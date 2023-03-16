package com.vvs.multithread;

import java.util.concurrent.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:06 2020/10/13
 * @Modified By:
 */
public class CountDownLatchTest {

    public static void main(String[] args) {
        //final CountDownLatch countDownLatch = new CountDownLatch(2);
        System.out.println("start.....");

        ThreadPoolExecutor executor1 = new ThreadPoolExecutor(2, 3, 60L, TimeUnit.SECONDS,new LinkedBlockingQueue<>(5), Executors.defaultThreadFactory(), new ThreadPoolExecutor.AbortPolicy());
        executor1.execute(new Runnable() {
            @Override
            public void run() {
//                    Thread.sleep(3000);
                    System.out.println("子线程：" +  Thread.currentThread().getName() + "执行");

                //countDownLatch.countDown();
            }
        });
        executor1.shutdown();

        ThreadPoolExecutor executor2 = new ThreadPoolExecutor(5, 10, 60L, TimeUnit.SECONDS,new LinkedBlockingQueue<>(5), Executors.defaultThreadFactory(), new ThreadPoolExecutor.AbortPolicy());
        executor2.execute(new Runnable() {
            @Override
            public void run() {
//                    Thread.sleep(3000);
                    System.out.println("子线程：" +  Thread.currentThread().getName() + "执行");

                //countDownLatch.countDown();
            }
        });
        executor2.shutdown();
        System.out.println("线程执行完毕......");

//        try {
//            countDownLatch.await();
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
        System.out.println("子线程结束， 执行主线程....");
    }
}
