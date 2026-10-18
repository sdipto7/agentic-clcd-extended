package com.company;

import JavaParser.JavaLexer;
import JavaParser.JavaParser;
import PythonParser.PythonLexer;
import PythonParser.PythonParser;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

public class PairFeatureExtractor {
    public static void main(String[] args) throws IOException {
        if (args.length != 3) {
            System.err.println("usage: java com.company.PairFeatureExtractor <java|python> <input_dir> <output_csv>");
            System.exit(2);
        }

        String language = args[0].toLowerCase(Locale.ROOT);
        Path inputDir = Paths.get(args[1]);
        Path outputCsv = Paths.get(args[2]);
        String suffix = language.equals("java") ? ".java" : ".py";

        List<Path> files = new ArrayList<>();
        try (Stream<Path> walk = Files.walk(inputDir)) {
            walk.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().endsWith(suffix))
                    .sorted(Comparator.comparing(path -> path.getFileName().toString()))
                    .forEach(files::add);
        }

        Files.createDirectories(outputCsv.toAbsolutePath().getParent());
        try (BufferedWriter writer = Files.newBufferedWriter(outputCsv, StandardCharsets.UTF_8)) {
            writer.write("pair_id,f1_variables,f2_arguments,f3_operators,f4_expressions,f5_loops,f6_operands,f7_exceptions,f8_exception_clauses,f9_mccabe");
            writer.newLine();
            for (Path file : files) {
                String pairId = stripExtension(file.getFileName().toString());
                int[] features = language.equals("java") ? javaFeatures(file) : pythonFeatures(file);
                writer.write(pairId);
                for (int feature : features) {
                    writer.write(",");
                    writer.write(Integer.toString(feature));
                }
                writer.newLine();
            }
        }
    }

    private static String stripExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot == -1 ? name : name.substring(0, dot);
    }

    private static int[] javaFeatures(Path path) throws IOException {
        CharStream input = CharStreams.fromFileName(path.toAbsolutePath().toString());
        JavaLexer lexer = new JavaLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        JavaParser parser = new JavaParser(tokens);
        ParseTree tree = parser.compilationUnit();

        ParseTreeWalker walker = new ParseTreeWalker();
        Listener listener = new Listener(parser);
        walker.walk(listener, tree);

        return new int[]{
                listener.getVariableno(),
                listener.getNoofarguments(),
                listener.getNoofoperators(),
                listener.getNoofexpression(),
                listener.getNoofloops(),
                listener.getNoofoperands(),
                listener.getNoofexceptions(),
                listener.getNoofexceptionclause(),
                listener.getMccabecomplex()
        };
    }

    private static int[] pythonFeatures(Path path) throws IOException {
        CharStream input = CharStreams.fromFileName(path.toAbsolutePath().toString());
        PythonLexer lexer = new PythonLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        PythonParser parser = new PythonParser(tokens);
        ParseTree tree = parser.root();

        ParseTreeWalker walker = new ParseTreeWalker();
        PythonListener listener = new PythonListener(parser);
        walker.walk(listener, tree);

        return new int[]{
                listener.getNoOfVariables(),
                listener.getNoOfArguments(),
                listener.getNoOfOperators(),
                listener.getNoOfExpressions(),
                listener.getNoOfLoops(),
                listener.getNoOfOperands(),
                listener.getNoOfExceptions(),
                listener.getNoOfExceptionClauses(),
                listener.getMccabeComplexity()
        };
    }
}
