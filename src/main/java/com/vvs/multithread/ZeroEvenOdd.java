package com.vvs.multithread;

import java.awt.print.Book;
import java.util.concurrent.Semaphore;
import java.util.function.IntConsumer;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:58 2020/8/30
 * @Modified By:
 */
class ZeroEvenOdd {
    private int n;
    private Object lock = new Object();
    private boolean zeroflag = true;
    private boolean evenflag;
    private boolean oddflag;
    private boolean flag;

    public ZeroEvenOdd(int n) {
        this.n = n;
    }

    // printNumber.accept(x) outputs "x", where x is an integer.
    public void zero(IntConsumer printNumber) throws InterruptedException {
        synchronized (lock){
            for(int i = 1;i <= n; i++){
                while(!zeroflag){
                    lock.wait();
                }
                printNumber.accept(0);
                if(flag == false){
                    oddflag = true;
                }else{
                    evenflag = true;
                }
                zeroflag = true;
                lock.notifyAll();
            }
        }
    }

    public void even(IntConsumer printNumber) throws InterruptedException {
        synchronized (lock){
            for(int i = 2;i <= n; i+=2){
                while(!evenflag){
                    lock.wait();
                }
                printNumber.accept(i);
                zeroflag = true;
                evenflag = false;
                flag = false;
                lock.notifyAll();
            }
        }
    }

    public void odd(IntConsumer printNumber) throws InterruptedException {
        synchronized (lock){
            for(int i = 1;i <= n; i+=2){
                while(!oddflag){
                    lock.wait();
                }
                printNumber.accept(i);
                zeroflag = true;
                oddflag = false;
                flag = true;
                lock.notifyAll();
            }
        }
    }
}