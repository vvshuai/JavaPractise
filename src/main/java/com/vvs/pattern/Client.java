package com.vvs.pattern;

import java.io.IOException;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 14:58 2020/7/1
 * @Modified By:
 */
public class Client {

    public static void main(String[] args) {
        AbstarctPrototype prototype = new PrototypeImpl("vvs");
        AbstarctPrototype clone = prototype.myClone();
        System.out.println(clone.toString());
        Runtime runtime = Runtime.getRuntime();
        try {
            runtime.exec("java -version");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
