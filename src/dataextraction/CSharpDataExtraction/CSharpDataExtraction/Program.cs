using CsvHelper;
using CsvHelper.Configuration.Attributes;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Threading.Tasks;

namespace CSharpDataExtraction
{
    class Program
    {
        private static readonly int URL_NUM_LEADING_ZEROS = 12;

        static void Main(string[] args)
        {
            if (args.Length != 3)
            {
                Console.WriteLine("Usage: <baseURL> <numChunksToRead> <ouputFilePrefix>");
                return;
            }
            var baseUrl = args[0];
            var numChunksToRead = int.Parse(args[1]);

            var outputFilePrefix = args[2];
            Directory.CreateDirectory(Path.GetDirectoryName(outputFilePrefix));

            using (var writer = new ChunkedJsonGzWriter(
                outputFilenameTemplate: outputFilePrefix,
                useJsonlFormat: true))
            {
                var extractor = new RepoDataExtractor();
                void ExtractRecord(CodeRecord rec)
                {
                    var parts = rec.RepositoryPath.Split();
                    extractor.ExtractDataFrom(parts[0], parts[1], rec.Content, writer);
                };

                Parallel.ForEach(
                    source: EnumerateCsvRecords(baseUrl, numChunksToRead),
                    body: ExtractRecord
                 );
            }
        }

        private static IEnumerable<CodeRecord> EnumerateCsvRecords(string baseUrl, int numChunksToRead)
        {
            for (int i = 0; i < numChunksToRead; i++)
            {
                var id = i.ToString("D" + URL_NUM_LEADING_ZEROS);
                var uri = $"{baseUrl}{id}.csv";
                Console.WriteLine($"Downloading {uri}...");

                using (var downloader = new WebClient())
                {
                    using (var stream = new StreamReader(downloader.OpenRead(uri)))
                    using (var csv = new CsvReader(stream))
                    {
                        csv.Configuration.HasHeaderRecord = true;
                        csv.Configuration.PrepareHeaderForMatch = (string header, int index) => header;
                        foreach (var record in csv.GetRecords<CodeRecord>())
                        {
                            yield return record;
                        }
                    }
                }
            }
        }
    }

    class CodeRecord
    {
        [Name("repo_path")]
        public string RepositoryPath { get; set; }

        [Name("content")]
        public string Content { get; set; }
    }
}
