package com.vvs.jvm0609;

import org.openjdk.jol.info.ClassLayout;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 16:31 2020/7/17
 * @Modified By:
 */
public class HelloJOL {

    public static void main(String[] args) {
        Object o = new Object();
        String s = ClassLayout.parseInstance(o).toPrintable();
        System.out.println(s);

        System.out.println("=====================");

        synchronized (o){
            System.out.println(ClassLayout.parseInstance(o).toPrintable());
        }
    }
}
