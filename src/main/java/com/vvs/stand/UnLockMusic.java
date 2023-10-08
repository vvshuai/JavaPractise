package com.vvs.stand;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;

import java.io.*;
import java.lang.Runtime;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class UnLockMusic {

    private static final String DIR = "D:\\old-music";

    private static final ExecutorService executorService = Executors.newFixedThreadPool(5);

    public static void main(String[] args) throws IOException, InterruptedException {
        File dirs = new File(DIR);
        File[] files = dirs.listFiles();
        assert files != null;
        for (File file : files) {
            String fileName = file.getName();
            if (fileName.contains(".mgg")) {
                fileName = file.getAbsolutePath();
                String s = "um.exe -o D:\\old-music\\output -i " + "\"" + fileName + "\"";
                Process process = Runtime.getRuntime().exec(s);
                InputStream inputStream = process.getInputStream();
                executorService.execute(() -> printStream(inputStream, "Error"));
                executorService.execute(() -> printStream(process.getErrorStream(), "Input"));
            }
        }
    }

    public static void printStream(final InputStream inputStreamData, final String type) {
        try (final InputStream inputStream = inputStreamData) {
            final StringWriter writer = new StringWriter();
            IOUtils.copy(inputStream, writer, StandardCharsets.UTF_8.name());
            if (type.equals("Error")) {
                System.out.println(writer.toString());
            } else {
                System.out.println(writer.toString());
            }
        } catch (final IOException e) {
            //不影响原有流程
        }
    }

}
