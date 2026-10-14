$server = "10.225.30.26"
$database = "LAMI_DB"
$diagramId = 1
$outputFile = "C:\Temp\production_operations.bin"

$username = Read-Host "Enter SQL Server username"
$password = Read-Host "Enter SQL Server password" -AsSecureString

$credential = New-Object System.Management.Automation.PSCredential($username, $password)

$plainPassword = $credential.GetNetworkCredential().Password

$connectionString = "Server=$server;Database=$database;User ID=$username;Password=$plainPassword;TrustServerCertificate=True"

$connection = New-Object System.Data.SqlClient.SqlConnection
$connection.ConnectionString = $connectionString

try {
    $connection.Open()

    Write-Host "Connected successfully."

    $command = $connection.CreateCommand()
    $command.CommandText = @"
SELECT definition
FROM dbo.sysdiagrams
WHERE diagram_id = $diagramId
"@

    $bytes = $command.ExecuteScalar()

    if ($null -eq $bytes) {
        Write-Host "Diagram not found."
    }
    else {
        [System.IO.Directory]::CreateDirectory("C:\Temp") | Out-Null
        [System.IO.File]::WriteAllBytes($outputFile, $bytes)

        Write-Host ""
        Write-Host "SUCCESS!"
        Write-Host "Diagram: production operations"
        Write-Host "Diagram ID: 1"
        Write-Host "Size: $($bytes.Length) bytes"
        Write-Host "Saved to: $outputFile"
    }
}
catch {
    Write-Host ""
    Write-Host "ERROR:"
    Write-Host $_.Exception.Message
}
finally {
    if ($connection.State -eq "Open") {
        $connection.Close()
    }
}