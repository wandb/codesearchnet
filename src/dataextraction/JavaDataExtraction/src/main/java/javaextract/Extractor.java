package javaextract;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.io.input.BOMInputStream;

import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import com.github.javaparser.JavaParser;
import com.github.javaparser.JavaToken;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.TokenRange;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.JavadocComment;
import com.github.javaparser.javadoc.Javadoc;


public class Extractor {

    private static final int MIN_SUMMARY_LENGTH = 3;
    private static final int MIN_NUM_LINES = 3;

    private static final int URL_NUM_LEADING_ZEROS = 12;

    public static void main(String[] args) throws IOException {
        // Usage <baseURL> <numChunks> <ouputFilePrefix>
        ChunkWriter<MethodData> chunkWriter = new ChunkWriter<>(args[2], 5000);

        String baseUrl = args[0];
        int numFiles = Integer.parseInt(args[1]);
        try {
            for (int i = 0; i < numFiles; i++) {
                URL url = new URL(String.format("%s%0"+ URL_NUM_LEADING_ZEROS + "d.csv", baseUrl, i));

                System.out.println("Downloading "+ url + "...");

                final Reader reader = new InputStreamReader(new BOMInputStream(url.openStream()), "UTF-8");
                final CSVParser parser = new CSVParser(reader, CSVFormat.EXCEL.withHeader());
                try {
                    StreamSupport.stream(parser.spliterator(), /* parallel= */ true).forEach(record->
                    {
                        String fileInfo = record.get("repo_path");
                        String content = record.get("content");
                        ExtractAllFromFile(fileInfo, content).forEach(chunkWriter::add);
                    });
                } finally {
                    parser.close();
                    reader.close();
                }
            }
        } finally {
            chunkWriter.close();
        }
    }

    public static List<MethodData> ExtractAllFromFile(String fileInfo, String fileContent) {
        List<MethodData> allMethods = new ArrayList<>();
        CompilationUnit cu;
        try {
            cu = JavaParser.parse(fileContent);
        } catch (ParseProblemException e) {
            System.err.println("Failed to parse " + fileInfo);
            return allMethods;
        }

        MethodFinder finder = new MethodFinder();
        cu.accept(finder, null);
        // Find all the calculations with two sides:
        for (MethodDeclaration decl: finder.allDeclarations) {
            if (!decl.getBody().isPresent()){
                 continue;  // ignore interfaces
            }
            if (!decl.hasJavaDocComment()) {
                continue;
            }
            if (decl.getNameAsString().toLowerCase().contains("test")) {
                continue;
            }

            MethodData methodData = new MethodData();

            methodData.code = decl.toString();

            methodData.func_name = decl.getNameAsString();
            methodData.lineno = decl.getRange().get().begin.line;

            if (decl.getRange().get().end.line - decl.getRange().get().begin.line + 1 <= MIN_NUM_LINES) {
                continue;
            }

            String[] info = fileInfo.split(" ");
            methodData.repo = info[0];

            methodData.path = String.join(" ", Arrays.copyOfRange(info, 1, info.length));

            // JavaDoc
            JavadocComment comment = decl.getJavadocComment().get();
            Javadoc javadoc = JavaParser.parseJavadoc(comment.toString());

            methodData.docstring = javadoc.getDescription().toText().replace("/**", "").trim();
            methodData.docstring_tokens = tokenizeComment(methodData.docstring);
            if (methodData.docstring_tokens.size() < MIN_SUMMARY_LENGTH) {
                continue;
            }

            TokenRange tokenRange =  decl.getTokenRange().get();
            methodData.code_tokens = new ArrayList<>();
            List<String> commentPseudotokens = new ArrayList<>();
            for (JavaToken token : tokenRange) {
                if (!token.getNextToken().isPresent()) {
                    continue;
                }
                if (IsCodeTokenKind(token)) {
                    methodData.code_tokens.add(token.getText());
                } else if (IsComment(token)) {
                    commentPseudotokens.add(token.getText().replace("//", "").trim());
                }
            }

            methodData.comment_tokens = commentPseudotokens.stream()
                                            .flatMap(m->tokenizeComment(m).stream())
                                            .collect(Collectors.toList());
            allMethods.add(methodData);
        }
        return allMethods;
    }

    private static boolean IsCodeTokenKind(JavaToken token) {
        switch(JavaToken.Kind.valueOf(token.getKind())) {
            case SPACE:
            case EOF:
            case UNIX_EOL:
            case WINDOWS_EOL:
            case OLD_MAC_EOL:

            case ENTER_JAVADOC_COMMENT:
            case ENTER_MULTILINE_COMMENT:
            case JAVADOC_COMMENT:
            case MULTI_LINE_COMMENT:
            case SINGLE_LINE_COMMENT:
            case COMMENT_CONTENT:
                return false;
            default:
                return true;
        }
    }

    private static boolean IsComment(JavaToken token) {
        switch(JavaToken.Kind.valueOf(token.getKind())) {
            case ENTER_MULTILINE_COMMENT:
            case MULTI_LINE_COMMENT:
            case SINGLE_LINE_COMMENT:
            case COMMENT_CONTENT:
                return true;
            default:
                return false;
        }
    }

    private static List<String> tokenizeComment(String comment) {
        return Arrays.asList(comment.split("[\\s()\\[\\].@\"'/,?]+"));
    }
}
