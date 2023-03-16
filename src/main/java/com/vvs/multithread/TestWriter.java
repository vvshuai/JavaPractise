package com.vvs.multithread;

import java.io.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 17:56 2021/12/4
 * @Modified By:
 */
public class TestWriter {

    public static void main(String[] args) throws IOException {

        BufferedReader br = new BufferedReader( new InputStreamReader(System.in) );

        System.out.println("Enter the file name to read");
        String fileName = br.readLine();
        br.close();

// Process file contents

        br = new BufferedReader( new InputStreamReader(System.in) );
        System.out.println("Enter another file name to read");
        fileName = br.readLine();
        br.close();
    }
}
