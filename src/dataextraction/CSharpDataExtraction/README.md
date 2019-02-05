To extract C# data, we assume that there is an input folder that contains multiple git repositories. Run
```
$ dotnet run path/to/CSharpDataExtraction.csproj https://storage.googleapis.com/kubeflow-examples/code_search_new/csharp_raw/ <numCsvChunks> <outputPath>
```
