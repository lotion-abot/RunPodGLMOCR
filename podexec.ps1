# =============================================================================
# podexec.ps1 - reliable command/file channel to a RunPod pod over ssh.runpod.io
#
#   . C:\RunPodGLMOcr\podexec.ps1
#   Invoke-Pod  'nvidia-smi; df -h /'
#   Send-PodFile C:\RunPodGLMOcr\pod_setup.sh /workspace/pod_setup.sh
#
# WHY THIS EXISTS
# ---------------
# The ssh.runpod.io proxy refuses non-PTY sessions ("Your SSH client doesn't
# support PTY") AND ignores an exec command, so `ssh host "cmd"` just hangs, and
# scp/sftp are unsupported on the proxy. The only working channel is an
# interactive PTY shell.
#
# Typing payloads into an interactive bash means readline: bracketed paste, tab
# completion and the MOTD warm-up mangled our first attempt (`uname -a` arrived
# as `ame -a`). So we never type a payload at the prompt. We type ONE short
# command - `cat > file` - after which bash is out of the loop and bytes go
# straight into cat. EOF is a literal Ctrl-D. The payload is base64 (no
# metacharacters, no tabs) wrapped at 200 chars, because a PTY in canonical mode
# truncates input lines past ~4096.
#
# Every file transfer is md5-verified against the local file. Corruption is
# detected, never silently accepted.
#
# NOTE: deliberately no here-strings and pure ASCII - PowerShell 5.1 choked on
# them here and debugging that was burning pod-hours.
# =============================================================================

$script:PodUser = "7flxvv3va0w9aj-64410f06@ssh.runpod.io"
$script:PodKey  = "$env:USERPROFILE\.ssh\id_ed25519"
$script:EOT     = [string][char]4

function Invoke-PodRaw {
    # Drives ssh through a real Process so we can PACE stdin. A PTY in canonical
    # mode has a ~4096-byte input buffer; dumping 10KB of base64 at it in one go
    # silently DROPS characters. We write a few lines at a time with a short
    # pause so the remote `cat` keeps up.
    param([string[]]$Lines, [int]$TimeoutSec = 300)

    # 4 blank lines up front absorb the MOTD / readline warm-up that ate
    # characters off our very first command; anything lost lands on an empty line.
    $all = @("", "", "", "") + $Lines + @("exit")

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "ssh"
    $psi.Arguments = "-tt -o BatchMode=yes -o StrictHostKeyChecking=accept-new " +
                     "-o ConnectTimeout=20 -o ServerAliveInterval=30 " +
                     "-i `"$script:PodKey`" $script:PodUser"
    $psi.RedirectStandardInput  = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow  = $true

    $p = [System.Diagnostics.Process]::Start($psi)
    $sbOut = New-Object System.Text.StringBuilder
    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()

    # SYNC BARRIER. Measured failure: anything written before the shell settles is
    # swallowed by the MOTD and by bash's completion pager ("Display all 1096
    # possibilities?" + --More--), which ate the first base64 lines and dropped
    # characters out of them. Every line sent AFTER the prompt settled arrived
    # intact. So: stay quiet, poke once, stay quiet again, then send.
    Start-Sleep -Seconds 6
    $p.StandardInput.Write("`n")
    $p.StandardInput.Flush()
    Start-Sleep -Seconds 3

    # MEASURED: exactly ONE line gets swallowed at the start of every session, no
    # matter how long we wait first. So burn two no-op lines as sacrifices before
    # anything that matters. (`:` is the bash no-op builtin.)
    # stty -echo then removes readline's echo of every line back over the network.
    foreach ($warm in @(": warmup", ": warmup", "stty -echo")) {
        $p.StandardInput.Write($warm + "`n")
        $p.StandardInput.Flush()
        Start-Sleep -Milliseconds 400
    }

    $n = 0
    foreach ($line in $all) {
        $p.StandardInput.Write($line + "`n")
        $n++
        if ($n % 2 -eq 0) { $p.StandardInput.Flush(); Start-Sleep -Milliseconds 200 }
    }
    $p.StandardInput.Flush()

    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        try { $p.Kill() } catch {}
        Write-Warning "ssh timed out after $TimeoutSec s - partial output below"
    }
    [void]$sbOut.Append($outTask.Result)
    [void]$sbOut.Append($errTask.Result)
    return $sbOut.ToString()
}

function Clear-Ansi {
    param([string]$Text)
    $esc = [char]27
    $bel = [char]7
    $t = $Text -replace "$esc\[[0-9;?]*[a-zA-Z]", ""
    $t = $t -replace "$esc\][^$bel$esc]*($bel|$esc\\)", ""
    $t = $t -replace "\?2004[hl]", ""
    $t = $t -replace "`r", ""
    return $t
}

function Split-B64 {
    # 100 chars/line. 200 overflowed the PTY's canonical input buffer under echo
    # backpressure and produced a DIFFERENT corrupt md5 on each of 3 attempts.
    param([string]$B64)
    $out = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $B64.Length; $i += 100) {
        $out.Add($B64.Substring($i, [Math]::Min(100, $B64.Length - $i)))
    }
    return $out.ToArray()
}

function Compress-Bytes {
    # gzip before base64: ~3x less to push through the PTY, which is the scarce
    # resource here. Pod side undoes it with `gzip -d`.
    param([byte[]]$Bytes)
    $ms = New-Object System.IO.MemoryStream
    $gz = New-Object System.IO.Compression.GZipStream($ms, [System.IO.Compression.CompressionMode]::Compress)
    $gz.Write($Bytes, 0, $Bytes.Length)
    $gz.Close()
    return $ms.ToArray()
}

function Get-PodBody {
    param([string]$Clean)
    $b = $Clean.IndexOf("__BEGIN__")
    $e = $Clean.LastIndexOf("__END__")          # last one = the echoed result
    if ($b -lt 0 -or $e -lt $b) { return $null }
    $body = $Clean.Substring($b + 9, $e - $b - 9)
    # drop the line where bash echoed our own trailing `echo __END__...`
    return (($body -split "`n" | Where-Object { $_ -notmatch '^\s*echo __END__' }) -join "`n")
}

function Get-ShipLines {
    # Builds the pod-side lines that reconstruct $Bytes at $RemotePath.
    #
    # We do NOT use `cat > file` + Ctrl-D any more. Measured: the payload bytes
    # arrive intact (bash echoed them back verbatim) but the `cat >` line itself
    # kept getting swallowed by the shell warm-up, so every base64 line then ran
    # as a command. What DOES survive reliably is an ordinary command line. So
    # each chunk is its own self-contained `echo -n '...' >>` append. base64's
    # alphabet (A-Za-z0-9+/=) is inert inside single quotes.
    param([byte[]]$Bytes, [string]$RemotePath)

    $b64 = [Convert]::ToBase64String((Compress-Bytes $Bytes))
    $rdir = ($RemotePath -replace '/[^/]+$', '')
    $out = New-Object System.Collections.Generic.List[string]
    $out.Add("mkdir -p $rdir")
    # First chunk TRUNCATES with '>'; the rest append. Belt and braces: even if the
    # truncate line were lost, we would never silently append to a stale file the
    # way an `rm -f` on its own line did (that produced a 14252-byte /tmp/_ship.b64
    # accumulated across failed attempts, and a "not in gzip format" every time).
    $first = $true
    foreach ($chunk in (Split-B64 $b64)) {
        $op = if ($first) { ">" } else { ">>" }
        $out.Add("echo -n '$chunk' $op /tmp/_ship.b64")
        $first = $false
    }
    $out.Add("base64 -d /tmp/_ship.b64 > /tmp/_ship.gz && gzip -dc /tmp/_ship.gz > $RemotePath")
    return $out.ToArray()
}

function Invoke-Pod {
    # Runs a bash script on the pod, returns only its output. The script is
    # shipped as a file first, so quoting / newlines / $ are all safe.
    param([Parameter(Mandatory)][string]$Script)

    $bytes = [Text.Encoding]::UTF8.GetBytes(($Script -replace "`r`n", "`n"))
    $lines = (Get-ShipLines -Bytes $bytes -RemotePath "/tmp/_c.sh") + @(
        "echo __BEGIN__",
        "bash /tmp/_c.sh 2>&1",
        "echo __END__`$?"
    )
    $clean = Clear-Ansi (Invoke-PodRaw -Lines $lines)
    $body = Get-PodBody $clean
    if ($null -eq $body) {
        Write-Warning "sentinels not found - raw output follows"
        return $clean
    }
    $code = (($clean.Substring($clean.LastIndexOf("__END__")) -split "`n")[0]) -replace "__END__", ""
    return ($body.Trim() + "`n[exit " + $code.Trim() + "]")
}

function Send-PodFile {
    # Ships a local file to the pod, verifies md5, retries once on mismatch.
    param([Parameter(Mandatory)][string]$Local, [Parameter(Mandatory)][string]$Remote)

    $bytes = [IO.File]::ReadAllBytes($Local)
    $md5   = (Get-FileHash $Local -Algorithm MD5).Hash.ToLower()

    foreach ($attempt in 1..3) {
        $lines = (Get-ShipLines -Bytes $bytes -RemotePath $Remote) + @(
            "echo __BEGIN__",
            "md5sum $Remote | cut -d' ' -f1",
            "echo __END__0"
        )
        $body = Get-PodBody (Clear-Ansi (Invoke-PodRaw -Lines $lines))
        $got = $null
        if ($body) {
            $got = ($body -split "`n" | ForEach-Object { $_.Trim() } |
                    Where-Object { $_ -match '^[0-9a-f]{32}$' } | Select-Object -First 1)
        }
        if ($got -eq $md5) {
            Write-Host ("  SENT ok  " + $Local + " -> " + $Remote + "  (" + $bytes.Length + " bytes, md5 " + $md5 + ")")
            return $true
        }
        Write-Warning ("  md5 mismatch attempt " + $attempt + ": local=" + $md5 + " remote=" + $got)
    }
    throw ("Send-PodFile FAILED (md5 mismatch twice): " + $Local)
}

Write-Host ("podexec loaded - pod: " + $script:PodUser)
