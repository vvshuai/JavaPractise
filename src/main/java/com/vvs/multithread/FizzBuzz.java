package com.vvs.multithread;

import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntConsumer;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 21:56 2020/10/13
 * @Modified By:
 */
public class FizzBuzz {

    private int n;

    private int i = 1;
    private ReentrantLock lock = new ReentrantLock();


    public FizzBuzz(int n){
        this.n = n;
    }

    // printFizz.run() outputs "fizz".
    public void fizz(Runnable printFizz) throws InterruptedException {

    }

    // printBuzz.run() outputs "buzz".
    public void buzz(Runnable printBuzz) throws InterruptedException {

    }

    // printFizzBuzz.run() outputs "fizzbuzz".
    public void fizzbuzz(Runnable printFizzBuzz) throws InterruptedException {

    }

    // printNumber.accept(x) outputs "x", where x is an integer.
    public void number(IntConsumer printNumber) throws InterruptedException {

    }
}
