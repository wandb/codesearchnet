Java Data Extractor
-----

To rebuild the project, delete the target folder and run
```
mvn package
```

Convert Java code into a our . To run, use
```
mvn exec:java -DXmx8G -D"exec.mainClass"="javaextract.Extractor" -D"exec.args"="PATH_TO_INPUT_FOLDER NUM_CHUNKS OUTPUT_FILE_PREFIX"
```

for example, Miltos used on Dec 6, 2018:
```
mvn exec:java -DXmx4G -D"exec.mainClass"="javaextract.Extractor" -D"exec.args"="https://storage.googleapis.com/kubeflow-examples/code_search_new/java_raw/ 100 /mnt/c/Users/t-mialla/Downloads/javadata/javamethods"
```

To use the RosettaExtractor
```
mvn exec:java -DXmx8G -D"exec.mainClass"="javaextract.RosettaExtractor" -D"exec.args"="INPUT_FILE OUTPUT_FILE_PREFIX"
```

For example
```
mvn exec:java -DXmx8G -D"exec.mainClass"="javaextract.RosettaExtractor" -D"exec.args"="PATH_TO/java_rosetta_code.csv java_rosetta_data"
```
