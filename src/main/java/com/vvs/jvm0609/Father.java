package com.vvs.jvm0609;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:08 2020/11/16
 * @Modified By:
 */
public class Father {

    static {
        System.out.println("父类静态代码块");
    }

    {
        System.out.println("父类代码块");
    }

    public Father() {
        System.out.println("父类构造函数");
    }
}

class Son extends Father{

    static {
        System.out.println("子类静态代码块");
    }

    {
        System.out.println("子类代码块");
    }

    public Son() {
        System.out.println("子类代码块");
    }

    public static void main(String[] args) {
        Son son = new Son();
    }
}
