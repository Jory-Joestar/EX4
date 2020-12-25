package topcount;

import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;

public class TopMerchants {
        
    //这里的思路是直接将csv文件当作文本文件处理

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: JavaWordCount <input_file> <output_path>");
            System.exit(1);
        }
        String[] input_files = args[0].split(",");
        if (input_files.length<2) {
            System.err.println("Please give two input files include <user_log_format>,<user_info_format>");
            System.exit(1);
        }
        String user_log_file = input_files[0];

        SparkSession spark = SparkSession.builder().appName("TopMerchantsCount").getOrCreate();
        JavaRDD<String> log_lines = spark.read().textFile(user_log_file).javaRDD();
        JavaRDD<List<String>> log_words = log_lines.map(s -> Arrays.asList(s.split(",")));
        //先筛选出1111日，action!=0的log
        log_words = log_words.filter(
            s -> {
                String timestamp=s.get(5);
                String action_type=s.get(6);
                return timestamp.equals("1111") && !action_type.equals("0");
            }
        );
        //然后创建log_pair，key是user_id，value是mercant_id。
        JavaPairRDD<String,String> log_pairs = log_words.mapToPair(
            s -> {
                return new Tuple2<String,String>(s.get(0),s.get(3));
            }
        );

        String user_info_file = input_files[1];
        JavaRDD<String> info_lines = spark.read().textFile(user_info_file).javaRDD();
        JavaRDD<List<String>> info_words = info_lines.map(s -> Arrays.asList(s.split(",")));
        //创建info_pair，key是user_id，value是年龄
        info_words = info_words.filter(
            s -> {
                return s.size()>2;
            }
        );
        JavaPairRDD<String,String> info_pairs = info_words.mapToPair(
            s -> {
                return new Tuple2<String,String>(s.get(0),s.get(1));
            }
        );

        //然后将两个pair连接，形成PairRdd:<user_id,(merchant_id,age_range)>
        JavaPairRDD<String,Tuple2<String,String>> joined_logs = log_pairs.join(info_pairs);
        //筛选出年龄在(1,2,3)中的log
        List<String> youth = Arrays.asList(new String[] {"1", "2", "3"});
        joined_logs = joined_logs.filter(
            s -> {
                String age_range = s._2._2;
                return youth.contains(age_range);
            }
        );
        //再转化成<merchant_id, count>的key-value对，准备用来计数
        JavaPairRDD<String, Integer> ones = joined_logs.mapToPair(
            s -> {
                String merchant_id = s._2._1;
                return new Tuple2<String,Integer>(merchant_id,1);
            }
        );
        
        //后面的操作就和商品计数一样了
        JavaPairRDD<String, Integer> counts = ones.reduceByKey((i1, i2) -> i1 + i2);
        JavaPairRDD<String, Integer> sorted = counts
        .mapToPair(s -> new Tuple2<Integer, String>(s._2, s._1))
        .sortByKey(false)
        .mapToPair(s -> new Tuple2<String, Integer>(s._2, s._1));
        List<Tuple2<String, Integer>> output = sorted.take(100);

        //List<Tuple2<String, Integer>> output = counts.collect();
        int rank = 1;
        for (Tuple2<?, ?> tuple : output) {
            System.out.println(rank + ": " + tuple._1() + ", " + tuple._2());
            rank = rank+1;
        }
        //counts.saveAsTextFile(args[1]);
        sorted.zipWithIndex()
        .mapValues(x -> x+1)
        .filter(s -> s._2<=100)
        .mapToPair(s -> new Tuple2<Long, Tuple2<String, Integer>>(s._2, s._1))
        .saveAsTextFile(args[1]);
        spark.stop();
    }
}