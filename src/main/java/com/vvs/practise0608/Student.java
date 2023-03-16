package com.vvs.practise0608;

import java.text.SimpleDateFormat;
import java.time.Instant;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Scanner;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 15:57 2020/9/17
 * @Modified By:
 */
//请定义Person类
class Person{
    String name;
    int age;
    String nationality;
    private static int count = 0;

    public Person() {
        count++;
    }

    public Person(String name, int age, String nationality) {
        this.name = name;
        this.age = age;
        this.nationality = nationality;
        count++;
    }

    public void getName() {
        System.out.println("Name="+name+";");
    }

    public void getAge() {
        System.out.println("Age="+age+";");
    }

    public void getNationality() {
        System.out.println("Nationality="+nationality+";");
    }

    public void growUp() {
        age++;
    }

    public void show() {
        System.out.println("Person count:"+count+";");
    }
}
//请定义Nationality接口
interface Nationality{
    void custom();

    void policy();
}

//定义一个Student类继承Person类，实现Nationality接口

public class Student extends Person implements Nationality{

    private String record;

    public Student(String name, int age, String n, String postgraduate) {
        this.name = name;
        this.age = age;
        this.nationality = n;
        this.record = postgraduate;
    }

    public Student(String name, String record) {
        this.name = name;
        this.nationality = "han";
        this.record = record;
    }

    //请参考main函数的内容编写以上内容
    public static void main(String args[]) {
        Person a = new Person("zhangsan", 19, "han");
        a.getName();
        a.getAge();
        a.getNationality();
        a.growUp();
        a.getAge();
        a.show();

        Person b = new Person("lisi", 39, "man");
        b.getName();
        b.getAge();
        b.getNationality();
        b.growUp();
        b.getAge();
        b.show();

        Student c = new Student("wanger", 33, "man", "Postgraduate");
        c.growUp();
        c.custom();
        c.policy();
        c.show();

        Student d = new Student("wanger", "Undergraduate");
        d.growUp();
        d.custom();
        d.policy();
        d.show();
    }

    @Override
    public void growUp() {
        super.growUp();
        if(age < 23){
            System.out.println("Undergraduate;");
            record = "Undergraduate";
        }else{
            System.out.println("Postgraduate");
            record = "Postgraduate";
        }
    }

    @Override
    public void custom() {
        System.out.println("Spring Festival, Dragon Boat Festival, Mid-Autumn Festival");
    }

    @Override
    public void policy() {
        if(nationality.equals("han")){
            System.out.println("No preferential policies;");
        }else{
            System.out.println("Have preferential policies;");
        }
    }
}