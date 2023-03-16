package com.vvs.multithread;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 11:18 2021/12/4
 * @Modified By:
 */
public class StreamTest {

    static class SingletonHolder {
        static StreamTest instance = new StreamTest();
    }

    public static StreamTest getInstance() {
        return SingletonHolder.instance;
    }

    private static final vvsPoolExecutor vvsPoolExecutor = new vvsPoolExecutor();

    public static void main(String[] args) throws InterruptedException {
        List<String> cmdList = new ArrayList<>();
        cmdList.add("java -version");
        cmdList.add("ffmpeg -y -i D:\\ffmpeg\\v.mp4 -t 3 -c copy D:\\ffmpeg\\new.mp4");
        cmdList.add("netstat -ano");
        for (int i = 0; i < 3; i++) {
                StreamTest.getInstance().cmdProcess(cmdList.get(0));
        }
        vvsPoolExecutor.shutdown();
    }

    public void cmdProcess(String cmd) {
        Process process = null;
        try {
            process = Runtime.getRuntime().exec(cmd);
            SequenceInputStream sequenceInputStream = new SequenceInputStream(process.getInputStream(), process.getErrorStream());
            vvsPoolExecutor.addTask(this, "printStream", sequenceInputStream);
            sequenceInputStream.close();
            int waitValue = process.waitFor();
            if (waitValue == 0) {
                System.out.println("success");
            }
        } catch (Throwable e) {
            e.printStackTrace();
        } finally {
            if (process != null) {
                process.destroy();
            }
        }
    }

    public void printStream(final InputStream stream) {
        try (final InputStream inputStream = stream){
            Thread.sleep(100);
            final StringWriter writer = new StringWriter();
            IOUtils.copy(inputStream, writer, StandardCharsets.UTF_8);
            System.out.println(writer);
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }
}
