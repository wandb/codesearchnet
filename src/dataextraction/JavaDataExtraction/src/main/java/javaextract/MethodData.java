package javaextract;

import java.util.List;

public class MethodData {
    String code;
    List<String> code_tokens;
    String docstring;
    List<String> docstring_tokens;
    List<String> comment_tokens;
    
    String func_name;
    final String language = "java";
    int lineno;
    String path;
    String repo;
    String sha;
}