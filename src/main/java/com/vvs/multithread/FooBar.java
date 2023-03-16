package com.vvs.multithread;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:32 2020/8/30
 * @Modified By:
 */
class FooBar {

    private int n;
    private ReentrantLock lock = new ReentrantLock();
    private Condition foocondition = lock.newCondition();
    private Condition barcondition = lock.newCondition();
    private volatile int count = 1;

    public FooBar(int n) {
        this.n = n;
    }

    public void foo(Runnable printFoo) throws InterruptedException {

        for(int i = 0;i < n; i++){
            lock.lock();
            try{
                if(count != 1){
                    foocondition.await();
                }
                printFoo.run();
                barcondition.signal();
                count = 2;
            }finally {
                lock.unlock();
            }
        }
    }

    public void bar(Runnable printBar) throws InterruptedException {
        for (int i = 0; i < n; i++) {
            lock.lock();
            try{
                if(count != 2){
                    barcondition.await();
                }
                printBar.run();
                foocondition.signal();
                count = 1;
            }finally {
                lock.unlock();
            }
        }
    }

}
