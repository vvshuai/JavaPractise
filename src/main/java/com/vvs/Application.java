package com.vvs;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

import javax.swing.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 23:02 2020/9/1
 * @Modified By:
 */

@SpringBootApplication(scanBasePackages = {
        "com.vvs.weblearning"
})
@EnableScheduling
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
