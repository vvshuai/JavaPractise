package com.vvs.crawler;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 17:29 2020/6/16
 * @Modified By:
 */
public class CrawlerTest {
    public static void main(String[] args) {
        try {
            URL url = new URL("https://github.com/crisxuan/bestJavaer/blob/master/mypdf/java-all.pdf");
            File file = new File("D:/projects/JavaPractise/resources/crawler/bestJava-1.pdf");
            download(url,file);
        } catch (MalformedURLException e) {
            e.printStackTrace();
        }
    }

    private static void download(URL url,File file){
        FileOutputStream fos = null;
        InputStream is = null;
        try {
            URLConnection conn = url.openConnection();
            conn.connect();
            is = conn.getInputStream();
            byte[] bytes = new byte[10240];
            fos = new FileOutputStream(file);
            int temp = 0;
            while( (temp = is.read(bytes)) != -1){
                fos.write(bytes,0,temp);
            }
        } catch (MalformedURLException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally{
            try {
                if(is != null) {
                    is.close();
                }
                if(fos != null) {
                    fos.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


}
