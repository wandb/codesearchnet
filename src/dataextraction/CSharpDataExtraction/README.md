This is a good start to install dotnet.
https://dotnet.microsoft.com/learn/dotnet/hello-world-tutorial/intro

To extract C# data, we assume that there is an input folder that contains multiple git repositories. Run
```
$ dotnet run path/to/CSharpDataExtraction.csproj https://storage.googleapis.com/kubeflow-examples/code_search_new/csharp_raw/ <numCsvChunks> <outputPath>
```

To use RosettaCodeTokenExtractor for C# data
```
$ dotnet run --project path/to/RosettaCodeTokenExtractor.csproj path/to/RosettaCodeData/Lang/C-sharp/ path/to/C-sharp-tokens
```
