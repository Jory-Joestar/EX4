package topcount;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.sum;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.format_number;

import org.apache.spark.sql.SparkSession;

public class SqlCountAge {
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
        String user_info_file = input_files[1];
        SparkSession spark = SparkSession.builder().appName("SqlCount").getOrCreate();
        Dataset<Row> user_log = spark.read()
        .format("csv")
        .option("sep", ",")
        .option("header", true)
        .option("nullValue", "")
        .csv(user_log_file);
        user_log.createTempView("user_log");

        Dataset<Row> user_info = spark.read()
        .format("csv")
        .option("sep", ",")
        .option("header", true)
        .option("nullValue", "")
        .csv(user_info_file);
        user_info.createTempView("user_info");

        Dataset<Row> result = spark.sql(
            "select age_range, count(distinct ul.user_id) buy_num " +
            "from user_log ul " +
            "left join user_info ui " + 
            "on ul.user_id = ui.user_id " + 
            "where action_type=2 and time_stamp=1111 and age_range>=1 and age_range<=8 " +
            "group by age_range"
        );

        result = result.withColumn("percent",format_number(col("buy_num").divide( sum("buy_num").over()).multiply(100),5));
        result.show();
        result.javaRDD().saveAsTextFile(args[1]);
        spark.stop();

    }
}