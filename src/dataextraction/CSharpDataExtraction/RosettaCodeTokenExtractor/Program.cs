using System;
using System.IO;
using System.Linq;
using Microsoft.CodeAnalysis.CSharp;
using Newtonsoft.Json;

namespace RosettaCodeTokenExtractor
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length != 2)
            {
                Console.WriteLine("Usage: INPUT_FOLDER OUTPUT_FOLDER");
                return;
            }

            var inputFolder = args[0];
            var outputFolder = args[1];
            var numFiles = 0;

            using (var writer = new ChunkedJsonGzWriter(
                outputFilenameTemplate: Path.Combine(outputFolder, "csharp_rosetta_data"),
                useJsonlFormat: true))
            {
                var allFiles = Directory.EnumerateFiles(inputFolder, "*.cs", SearchOption.AllDirectories);

                foreach (var filePath in allFiles)
                {
                    writer.WriteElement(jw => ExtractTokenFrom(jw, filePath, inputFolder));
                    numFiles += 1;
                }
            }
            Console.WriteLine($"Processed {numFiles} files.");
        }

        private static void ExtractTokenFrom(JsonWriter jsonWriter, string filename, string basePath)
        {
            var syntaxTree = CSharpSyntaxTree.ParseText(File.ReadAllText(filename), path: filename);

            jsonWriter.WriteStartObject();

            jsonWriter.WritePropertyName("filename");
            jsonWriter.WriteValue(Path.GetRelativePath(basePath, filename));

            jsonWriter.WritePropertyName("tokens");
            jsonWriter.WriteStartArray();
            foreach(var token in syntaxTree.GetRoot().DescendantTokens().Where(t=>t.Text.Length > 0))
            {
                jsonWriter.WriteValue(token.Text);
            }
            jsonWriter.WriteEndArray();

            jsonWriter.WriteEndObject();
        }
    }
}
