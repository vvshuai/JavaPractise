package com.vvs.stand;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class TransformMusic {

    private static final String DIR = "D:\\old-music\\output";

    private static final String TRANS_DIR = "D:\\old-music\\trans\\";

    public static void main(String[] args) throws IOException {
        File dirs = new File(DIR);
        File[] files = dirs.listFiles();
        assert files != null;
        for (File file : files) {
            StringBuilder sb = new StringBuilder();

            File newFile = new File(TRANS_DIR + file.getName().replace(".ogg", ".mp3"));

            sb.append("ffmpeg -i ");
            sb.append("\"").append(file.getAbsoluteFile()).append("\"");
            sb.append(" -acodec libmp3lame ");
            sb.append("\"").append(newFile.getAbsoluteFile()).append("\"");

            Process exec = Runtime.getRuntime().exec(sb.toString());

            UnLockMusic.printStream(exec.getInputStream(), "Input");
            UnLockMusic.printStream(exec.getErrorStream(), "Error");
        }
    }
}
