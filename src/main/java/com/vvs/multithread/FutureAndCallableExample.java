package com.vvs.multithread;

import java.util.concurrent.*;

public class FutureAndCallableExample {

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        ExecutorService executorService = Executors.newSingleThreadExecutor();

        Callable<String> callable = () -> {
            System.out.println("Callable");
            Thread.sleep(5000);
            return "Hello callable";
        };

        Future<String> future = executorService.submit(callable);

        System.out.println("main thread running");
        System.out.println("main thread waiting future");
        String result = future.get();
        System.out.println("---" + result);
        executorService.shutdown();
    }
}
