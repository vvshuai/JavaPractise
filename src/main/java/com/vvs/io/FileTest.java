package com.vvs.io;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class FileTest {

    public static void main(String[] args) throws IOException {
        File file = new File("C:\\Users\\vvshuai\\Desktop\\1.txt");
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String s = "";
        List<Integer> list = new ArrayList<>();
        while ((s = reader.readLine()) != null ) {
            if (s.contains("resourcepath")) {
                int start = s.indexOf("resourcepath");
                s = s.substring(start + 35, start + 50);
                StringBuilder sb = new StringBuilder();
                for (int i = 0;i < s.length(); i++) {
                    if (s.charAt(i) == '.') {
                        break;
                    }
                    sb.append(s.charAt(i));
                }
                list.add(Integer.parseInt(sb.toString()));
            }
        }
        System.out.println(new ObjectMapper().writeValueAsString(list));
        System.out.println(list.size());
    }
}
