package topcount;

import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;


public class TopItems {

    //这里的思路是直接将csv文件当作文本文件处理

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: JavaWordCount <input_file> <output_path>");
            System.exit(1);
        }
        SparkSession spark = SparkSession.builder().appName("TopItemCount").getOrCreate();
        JavaRDD<String> lines = spark.read().textFile(args[0]).javaRDD();
        JavaRDD<List<String>> words = lines.map(s -> Arrays.asList(s.split(",")));
        JavaRDD<List<String>> useful = words.filter(
            s -> {
                String timestamp=s.get(5);
                String action_type=s.get(6);
                return timestamp.equals("1111") && !action_type.equals("0");
            }
        );
        JavaPairRDD<String, Integer> ones = useful.mapToPair(
            s -> {
                String item_id=s.get(1);
                return new Tuple2<>(item_id,1);
            }
        );
        JavaPairRDD<String, Integer> counts = ones.reduceByKey((i1, i2) -> i1 + i2);
        JavaPairRDD<String, Integer> sorted = counts
        .mapToPair(s -> new Tuple2<Integer, String>(s._2, s._1))
        .sortByKey(false)
        .mapToPair(s -> new Tuple2<String, Integer>(s._2, s._1));
        List<Tuple2<String, Integer>> output = sorted.take(100);;

        //List<Tuple2<String, Integer>> output = counts.collect();
        int rank = 1;
        for (Tuple2<?, ?> tuple : output) {
            System.out.println(rank + ": " + tuple._1() + ", " + tuple._2());
            rank = rank+1;
        }
        //sorted.saveAsTextFile(args[1]);
        sorted.zipWithIndex()
        .mapValues(x -> x+1)
        .filter(s -> s._2<=100)
        .mapToPair(s -> new Tuple2<Long, Tuple2<String, Integer>>(s._2, s._1))
        .saveAsTextFile(args[1]);
        spark.stop();
    }
    
}