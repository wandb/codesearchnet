using CSharpDataExtraction.Utils;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Newtonsoft.Json;
using System;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Xml;

namespace CSharpDataExtraction
{
    public class RepoDataExtractor
    {
        private const int MIN_SUMMARY_CHAR_LENGTH = 10;
        private const int MIN_NUM_LINES = 3;
        private readonly static ImmutableHashSet<string> BlackListedFunctionNames =
            ImmutableHashSet.CreateRange<string>(new[] { "test", "tests" });

        private static readonly Regex SplitCamelCaseRegex = new Regex(@"(?=\p{Lu}\p{Ll})|(?<=\p{Ll})(?=\p{Lu})|(?=_)|(?<=_)|((?<=\p{L})(?=[0-9]))|((?<=[0-9])(?=\p{L}))");
        
        public void ExtractDataFrom(string repoPath, string relativePath, string content, ChunkedJsonGzWriter writer)
        {
            var syntaxTree = CSharpSyntaxTree.ParseText(
                text: content, path: relativePath,
                options: CSharpParseOptions.Default.WithKind(SourceCodeKind.Script));
            ExtractDataFrom(syntaxTree, writer, repoPath);
        }

        private void ExtractDataFrom(SyntaxTree syntaxTree, ChunkedJsonGzWriter writer, string repoPath)
        {
            var compilation = CSharpCompilation.Create("tmpCompilation", syntaxTrees: new[] { syntaxTree });
            var compiledTree = compilation.SyntaxTrees.First();
            var semanticModel = compilation.GetSemanticModel(compiledTree);

            var allDeclaredMethods = MethodUtils.GetAllMethodDeclarations(compiledTree.GetRoot());
            foreach (var methodDeclarationNode in allDeclaredMethods.Where(m => m.Body != null))
            {
                try
                {
                    if (!(semanticModel.GetDeclaredSymbol(methodDeclarationNode) is IMethodSymbol methodSymbol))
                    {
                        continue;
                    }

                    if (SplitCamelCaseRegex.Split(methodDeclarationNode.Identifier.Text).Any(s => BlackListedFunctionNames.Contains(s.ToLower())))
                    {
                        continue;
                    }

                    var (summary, returns, parameters) = MethodUtils.GetDocumentationComment(methodSymbol);

                    // Replace <seealso cref="!:Fully.Qualified.Name#method()" /> tags with their cref content
                    // and other similar replacements
                    var summary_cleaned = Regex.Replace(summary, "</?[^\\n>]+/?>", new MatchEvaluator(ReplaceXml));

                    // If the summary has an empty line, remove everything beneath it.
                    var parts = Regex.Split(summary_cleaned, @"\n\s*\n").Select(p => p.Trim()).Where(p => p.Length > 0).ToArray();
                    if (parts.Length > 1)
                    {
                        summary_cleaned = parts[0];
                    }

                    if (string.IsNullOrWhiteSpace(summary_cleaned) || summary_cleaned.Length < MIN_SUMMARY_CHAR_LENGTH)
                    {
                        // Empty or too short summary
                        continue;
                    }

                    var lineSpan = compiledTree.GetMappedLineSpan(methodDeclarationNode.Body.Span);
                    if (lineSpan.EndLinePosition.Line - lineSpan.StartLinePosition.Line + 1 <= MIN_NUM_LINES)
                    {
                        continue; // Method seems to be too short.
                    }

                    writer.WriteElement(jw => WriteMethodData(methodDeclarationNode, summary, summary_cleaned, jw, repoPath));
                }
                catch (Exception e)
                {
                    Console.WriteLine($"Failed to extract data: {e.Message}");
                }
            }
        }

        private static string ReplaceXml(Match match)
        {
            var comment = match.Groups[0].Value;
            
            var xmlDoc = new XmlDocument();
            try
            {
                xmlDoc.LoadXml(comment);
                var element = xmlDoc.FirstChild;
                if (element.Name == "see" || element.Name == "seealso")
                {
                    var cref = element.Attributes["cref"].Value;
                    if (cref.IndexOf(":") != -1)
                    {
                        cref = cref.Substring(cref.IndexOf(":") + 1);
                    }
                    return cref;
                }
                else if (element.Name == "paramref")
                {
                    return element.Attributes["name"].Value;
                }
                return xmlDoc.InnerText;
            }
            catch (Exception)
            {
                return "";
            }
            
        }

        private void WriteMethodData(MethodDeclarationSyntax method, string summary, string summary_cleaned, JsonWriter jWriter, string repoPath)
        {
            jWriter.WriteStartObject();

            jWriter.WritePropertyName("code");
            jWriter.WriteValue(method.ToString());

            jWriter.WritePropertyName("code_tokens");
            jWriter.WriteStartArray();
            foreach (var token in method.DescendantTokens())
            {
                jWriter.WriteValue(token.ValueText);
            }
            jWriter.WriteEndArray();

            jWriter.WritePropertyName("docstring");
            jWriter.WriteValue(summary);

            jWriter.WritePropertyName("docstring_tokens");
            jWriter.WriteStartArray();
            foreach (var token in summary_cleaned
                .Split(new[] { ' ', '\n', '.', '(', ')', ';', '\'', '"', '@' },
                        StringSplitOptions.RemoveEmptyEntries)
                .Select(t => t.Trim()).Where(t => t.Length > 0))
            {
                jWriter.WriteValue(token);
            }
            jWriter.WriteEndArray();

            jWriter.WritePropertyName("comment_tokens");
            jWriter.WriteStartArray();
            foreach (var token in method.DescendantTrivia()
                .Where(t=>t.IsKind(SyntaxKind.SingleLineCommentTrivia) || t.IsKind(SyntaxKind.MultiLineCommentTrivia))
                .SelectMany(t=>t.ToString()
                    .Split(new[] { ' ', '\n', '.', '(', ')', ';', '\'', '"', '@', '/' },
                            StringSplitOptions.RemoveEmptyEntries))
                .Select(t => t.Trim()).Where(t => t.Length > 0))
            {
                jWriter.WriteValue(token);
            }
            jWriter.WriteEndArray();

            jWriter.WritePropertyName("func_name");
            jWriter.WriteValue(method.Identifier.ToString());

            jWriter.WritePropertyName("language");
            jWriter.WriteValue("csharp");

            jWriter.WritePropertyName("lineno");
            // Roslyn does 0-based line counting.
            jWriter.WriteValue(method.SyntaxTree.GetLineSpan(method.Span).StartLinePosition.Line + 1);

            jWriter.WritePropertyName("path");
            jWriter.WriteValue(method.SyntaxTree.FilePath);

            jWriter.WritePropertyName("repo");
            jWriter.WriteValue(repoPath);

            jWriter.WriteEndObject();
        }

    }
}
