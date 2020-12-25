package topcount;

import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CountAge {
    //统计双十一购买了商品的年龄比例
    /* 统计年龄比例，查询条件：user_log表左连接user_info表，
    日期=1111，action_type=2，按年龄分组，
    统计不重复的user_id数，还要考虑排除异常的age_range值。 */
    public static void main(String[] args) throws Exception  {
        //首先还是要做表的连接
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

        SparkSession spark = SparkSession.builder().appName("CountAge").getOrCreate();
        JavaRDD<String> log_lines = spark.read().textFile(user_log_file).javaRDD();
        JavaRDD<List<String>> log_words = log_lines.map(s -> Arrays.asList(s.split(",")));
        //先筛选出1111日，action!=0的log
        log_words = log_words.filter(
            s -> {
                String timestamp=s.get(5);
                String action_type=s.get(6);
                return timestamp.equals("1111") && action_type.equals("2");  //需要在1111日进行购买操作。
            }
        );
        //然后创建log_pair，key是user_id，value是item_id
        JavaPairRDD<String,String> log_pairs = log_words.mapToPair(
            s -> {
                return new Tuple2<String,String>(s.get(0),s.get(1));
            }
        );

        String user_info_file = input_files[1];
        JavaRDD<String> info_lines = spark.read().textFile(user_info_file).javaRDD();
        JavaRDD<List<String>> info_words = info_lines.map(s -> Arrays.asList(s.split(",")));
        //创建info_pair，key是user_id，value是年龄age_range
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

        //然后将两个pair连接，形成PairRdd:<user_id,(item_id,age_range)>
        JavaPairRDD<String,Tuple2<String,String>> joined_logs = log_pairs.join(info_pairs);
        //修改PairRdd的格式，改成<user_id,age_range>
        JavaPairRDD<String,String> user_age = joined_logs.mapToPair(
            s -> {
                return new Tuple2<String,String>(s._1,s._2._2);
            }
        );
        //去除重复的user_id
        user_age = user_age.reduceByKey(
            (x,y) -> {
                return x;
            }
        );
        //再调换PairRdd的key-value，改成<gender,1>，用来计数
        JavaPairRDD<String,Integer> age_ones = user_age.mapToPair(
            s -> {
                return new Tuple2<String,Integer>(s._2,1);
            }
        );
        //然后按年龄将计数加总
        JavaPairRDD<String, Integer> counts = age_ones.reduceByKey((i1, i2) -> i1 + i2);
        List<Tuple2<String, Integer>> output = counts.collect();
        List<Tuple2<String, Integer>> countable = new ArrayList<Tuple2<String, Integer>>();
        List<String> genders = Arrays.asList(new String[] {"1", "2", "3", "4", "5", "6", "7", "8"});
        int total_num = 0;
        for (Tuple2<String, Integer> tuple : output) {
            System.out.println(tuple._1() + ": " + tuple._2());
            if (genders.contains(tuple._1())) {
                countable.add(tuple);
                total_num += tuple._2();
            }
        }
        for (Tuple2<String, Integer> tuple : countable) {
            String num = String.format("%.1f", (double)tuple._2()/(double)total_num*100);
            System.out.println(tuple._1() + ": " + num + "%");
        }
        counts.saveAsTextFile(args[1]);
        spark.stop();
    }
}