using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace CSharpDataExtraction.Utils
{
    public static class GitInfo
    {
        public static string GetSHA(string gitDirectory)
        {
            var proc = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "git",
                    Arguments = "rev-parse --verify HEAD",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true,
                    WorkingDirectory = gitDirectory
                }
            };

            proc.Start();
            proc.WaitForExit();
            return proc.StandardOutput.ReadToEnd().Trim();
        }

        public static string GetRemote(string gitDirectory)
        {
            var proc = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "git",
                    Arguments = "config --get remote.origin.url",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true,
                    WorkingDirectory = gitDirectory
                }
            };

            proc.Start();
            proc.WaitForExit();
            return proc.StandardOutput.ReadToEnd().Trim();
        }
    }
}
