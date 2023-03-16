package com.vvs.jvm0609;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 21:02 2020/11/13
 * @Modified By:
 */
public class StaticDemo {

    public static void main(String[] args) {
        Runtime runtime = Runtime.getRuntime();
        String cmd = "java -version";
        Process cmdProcess = null;
        try {
            cmdProcess = runtime.exec(cmd);
            cmdProcess.waitFor();
            cmdProcess.destroy();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }

    }
}
