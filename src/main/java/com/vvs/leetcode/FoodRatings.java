package com.vvs.leetcode;

import java.util.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 10:40 2022/7/24
 * @Modified By:
 */
public class FoodRatings {

    Map<String, TreeSet<PPS>> map = new HashMap<>();
    Map<String, String> vv = new HashMap<>();
    Map<String, Integer> sc = new HashMap<>();

    public FoodRatings(String[] foods, String[] cuisines, int[] ratings) {
        for (int i = 0;i < foods.length; i++) {
            map.putIfAbsent(cuisines[i], new TreeSet<>((o1, o2) -> {
                if (!Objects.equals(o2.val, o1.val)) {
                    return o2.val - o1.val;
                }
                return o1.name.compareTo(o2.name);
            }));
            vv.put(foods[i], cuisines[i]);
            map.get(cuisines[i]).add(new PPS(foods[i], ratings[i]));
            sc.put(foods[i], ratings[i]);
        }
    }

    public void changeRating(String food, int newRating) {
        if (vv.get(food) != null && map.get(vv.get(food)) != null) {
            String s = vv.get(food);
            TreeSet<PPS> pps = map.get(s);
            PPS cur = new PPS(food, sc.get(food));
            pps.remove(cur);
            pps.add(new PPS(food, newRating));
            sc.put(food, newRating);
        }
    }

    public String highestRated(String cuisine) {
        return map.get(cuisine).first().name;
    }

    public static void main(String[] args) {
        FoodRatings foodRatings = new FoodRatings(new String[]{"tjokfmxg","xmiuwozpmj","uqklk","mnij","iwntdyqxi","cduc","cm","mzwfjk"},
                new String[]{"waxlau","ldpiabqb","ldpiabqb","waxlau","ldpiabqb","waxlau","waxlau","waxlau"}, new int[]{9,13,7,16,10,17,16,17});
        foodRatings.changeRating("tjokfmxg", 19);
        foodRatings.changeRating("tjokfmxg", 14);
        foodRatings.changeRating("tjokfmxg", 4);
    }

    class PPS {
        String name;
        Integer val;

        public PPS(String name, Integer val) {
            this.name = name;
            this.val = val;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            PPS pps = (PPS) o;
            return name.equals(pps.name) && val.equals(pps.val);
        }

        @Override
        public int hashCode() {
            return Objects.hash(name, val);
        }
    }
}
