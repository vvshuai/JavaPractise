package com.vvs.multithread;

import reactor.core.publisher.Mono;

import java.util.function.Consumer;

public class MonoTest {

    public static void main(String[] args) {
        Mono<byte[]> mono = Mono.empty();
        mono.subscribe(bytes -> {
            for (int i = 0;i < bytes.length; i++) {
                System.out.println(bytes[i]);
            }
            System.out.println("1");
        });
    }
}
