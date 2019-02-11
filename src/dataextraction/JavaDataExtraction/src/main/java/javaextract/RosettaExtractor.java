package javaextract;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.io.input.BOMInputStream;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.StreamSupport;

import com.github.javaparser.JavaParser;
import com.github.javaparser.JavaToken;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.TokenRange;
import com.github.javaparser.ast.CompilationUnit;


public class RosettaExtractor {

    public static void main(String[] args) throws IOException {
        // Usage <filename> <ouputFilePrefix>
        ChunkWriter<MethodData> chunkWriter = new ChunkWriter<>(args[1], 5000);

        String filename = args[0];
        try {
            URL url = new File(String.format(filename)).toURI().toURL();

            System.out.println("Reading "+ url + "...");

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

        MethodData methodData = new MethodData();
        String[] info = fileInfo.split(" ");
        methodData.repo = info[0];

        TokenRange tokenRange =  cu.getTokenRange().get();
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
        allMethods.add(methodData);
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

}
