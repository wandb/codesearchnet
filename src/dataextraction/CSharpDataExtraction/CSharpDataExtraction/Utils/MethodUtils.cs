using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace CSharpDataExtraction
{
    public static class MethodUtils
    {
        public static IEnumerable<MethodDeclarationSyntax> GetAllMethodDeclarations(SyntaxNode root)
        {
            return root.DescendantNodes().OfType<MethodDeclarationSyntax>();
        }

        public static (string Summary, string Returns, Dictionary<IParameterSymbol, string> ParameterComments)
            GetDocumentationComment(IMethodSymbol methodSymbol)
        {
            var comment = methodSymbol.GetDocumentationCommentXml().Trim();         

            if (string.IsNullOrWhiteSpace(comment))
            {
                return ("", "", null);
            }

            if (!comment.StartsWith("<member"))
            {
                comment = "<member>" + comment + "</member>";
            }
            var xmlDoc = new XmlDocument();
            try
            {
                xmlDoc.LoadXml(comment);
            } catch (Exception)
            {
                return ("", "", null);
            }

            if (xmlDoc.SelectSingleNode("member") == null)
            {
                return ("", "", null);
            }

            var memberXmlNode = xmlDoc.SelectSingleNode("member");

            string summary = "";
            if (memberXmlNode.SelectSingleNode("summary") != null) {
                summary = xmlDoc.SelectSingleNode("member").SelectSingleNode("summary").InnerXml.Trim();
            }

            var parameterComments = new Dictionary<IParameterSymbol, string>();
            var paramNamesToSymbols = methodSymbol.Parameters.GroupBy(s=>s.Name).ToDictionary(s => s.Key, s => s.First());

            foreach(var paramXmlNode in memberXmlNode.SelectNodes("param"))
            {
                var nameNode = ((XmlNode)paramXmlNode).Attributes["name"];
                if (nameNode == null) continue;
                var paramName = nameNode.InnerText;
                if (paramNamesToSymbols.ContainsKey(paramName))
                {
                    parameterComments[paramNamesToSymbols[paramName]] = ((XmlNode)paramXmlNode).InnerXml.Trim();
                }
            }

            string returnVal = "";
            if (memberXmlNode.SelectSingleNode("returns") != null)
            {
                returnVal = xmlDoc.SelectSingleNode("member").SelectSingleNode("returns").InnerXml.Trim();
            }
                
            return (summary, returnVal, parameterComments);            
        }     
    }
}
