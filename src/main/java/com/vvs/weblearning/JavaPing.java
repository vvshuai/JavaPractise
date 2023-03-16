package com.vvs.weblearning;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.time.LocalTime;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 0:15 2022/4/13
 * @Modified By:
 */
public class JavaPing {

    public static void main(String[] args) {
        String serverName = "124.221.69.230";
        int port = 9999;
        for (int i = 0;i < 10; i++) {
            LocalTime localTime = LocalTime.now();
            String s = localTime.toString();
            try {
                DatagramSocket socket = new DatagramSocket();
                byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
                DatagramPacket packet = new DatagramPacket(bytes, bytes.length, InetAddress.getByName(serverName), port);
                socket.send(packet);
                System.out.println("Message " + i + " Success");
                socket.close();
                Thread.sleep(1000);
            } catch (Exception e) {
                System.out.println("Message " + i + " Timeout");
            }
        }
    }
}
