Java Data Extractor
-----

Convert Java code into a our . To run, use
```
mvn exec:java -DXmx8G -D"exec.mainClass"="javaextract.Extractor" -D"exec.args"="PATH_TO_INPUT_FOLDER OUTPUT_FILE_PREFIX"
```

for example, Miltos used on Dec 6, 2018:
```
mvn exec:java -DXmx4G -D"exec.mainClass"="javaextract.Extractor" -D"exec.args"="https://storage.googleapis.com/kubeflow-examples/code_search_new/java_raw/ 100 /mnt/c/Users/t-mialla/Downloads/javadata/javamethods"
```