package com.vvs.weblearning;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 15:22 2022/4/4
 * @Modified By:
 */
public class WebFirst {

    public static void main(String[] args) throws IOException {
        ServerSocket serverSocket = new ServerSocket(6789);
        while (true) {
            Socket accept = serverSocket.accept();
            InputStream inputStream = accept.getInputStream();
            InputStreamReader reader = new InputStreamReader(inputStream);
            BufferedReader bufferedReader = new BufferedReader(reader);
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                if (line.equals("")) {
                    break;
                }
            }
            File file = new File("D:\\python\\webLearning\\hello.html");
            if (file.exists()) {
                FileInputStream fileInputStream = new FileInputStream(file);
                InputStreamReader reader1 = new InputStreamReader(fileInputStream);
                BufferedReader bufferedReader1 = new BufferedReader(reader1);
                PrintWriter out = new PrintWriter(accept.getOutputStream(), true);
                out.println("HTTP/1.1 200 OK");
                out.println("Connection: close");
                out.println("Content-Type: text/html");
                out.println("Content-Length: " + file.length());
                out.println("");
                while ((line = bufferedReader1.readLine()) != null) {
                    out.println(line);
                }
            }
            accept.close();
        }
    }
}
