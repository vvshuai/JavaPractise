package com.vvs.pattern;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 9:39 2020/9/11
 * @Modified By:
 */
public class Singleton {

    private static volatile Singleton instance;

    public static Singleton getInstance(){
        if(instance == null){
            synchronized (Singleton.class){
                if(instance == null){
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
