package com.vvs.multithread;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 21:37 2020/6/12
 * @Modified By:
 */
public class Basic {

    //ThreadLocal<T>
    public static ThreadLocal<Long> x = new ThreadLocal<Long>(){
        @Override
        protected Long initialValue() {
            System.out.println("Init...");
            return Thread.currentThread().getId();
        }
    };

    public static void main(String[] args){
        new Thread(){
            @Override
            public void run() {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println(x.get());
            }
        }.start();
        x.set(107L);
        x.remove();
        System.out.println(x.get());
    }
}
