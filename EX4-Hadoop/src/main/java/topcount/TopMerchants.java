package topcount;

import java.io.IOException;
import java.util.Random;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.InverseMapper;
import org.apache.hadoop.util.GenericOptionsParser;

public class TopMerchants {

    public static class JoinMapper extends Mapper<Object,Text,Text,Text> {

        String user_log_value="";

        public void map(Object key, Text value, Context context)
                throws IOException,InterruptedException {
            //根据命令行传入的文件名，判断数据来自哪个文件，来自user_log的数据打上a标签，来自user_info的数据打上b标签
            String filepath = ((FileSplit)context.getInputSplit()).getPath().toString();
            String line = value.toString();
            if (line == null || line.equals("")) return;

            if (filepath.indexOf("user_log_format1") != -1) {
                String[] words = line.split(",");
                String timestamp = words[5];
                String action_type = words[6];
                //可以先做一次筛选，选出日期为1111，同时action_type不为0的log
                if (timestamp.equals("1111") && !action_type.equals("0")) {
                    //key值为user_id
                    String user_id = words[0];
                    //value值只保存需要的信息：merchant_id，加上标签a
                    String merchant_id=words[3]+",a";
                    context.write(new Text(user_id), new Text(merchant_id));
                }
            } else if(filepath.indexOf("user_info_format1") != -1) {
                String[] words = line.split(",");
                if(words.length<2) return;
                //key值为user_id
                String user_id = words[0];
                //value值为年龄age_range，加上标签b
                String age_range = words[1]+",b";
                context.write(new Text(user_id), new Text(age_range));
            }
        }
    }

    public static class JoinReducer extends Reducer<Text, Text,NullWritable ,Text> {

        public List<String> youth = Arrays.asList(new String[] {"1", "2", "3"});

        public void reduce(Text key, Iterable<Text> values,
                            Context context) throws IOException, InterruptedException{
            List<String> merchant_list = new ArrayList<String>();
            String age_range = "";

            for(Text val:values) {
                String[] str = val.toString().split(",");
                //最后一位是标签位，因此根据最后一位判断数据来自哪个文件，
                //标签为a的数据为merchant_id，放在lista中；标签为b的数据即为年龄age_range
                String flag = str[str.length -1];
                if("a".equals(flag)) {
                    merchant_list.add(str[0]);
                } else if("b".equals(flag)) {
                    age_range=str[0];
                }
            }
            //在reduce阶段还可以进一步筛选，同时精简数据，只将年龄为30以下（age_range=1,2,3）购买的merchant_id写出。
            //下一个阶段的MapReduce只需要进行简单的WordCount即可。
            if (youth.contains(age_range)) {
                for (int i = 0; i < merchant_list.size(); i++) {
                    String merchant_id = merchant_list.get(i);
                    context.write(NullWritable.get(), new Text(merchant_id));
                }
            }
        }
    }

    public static class TokenizerMapper
    extends Mapper<NullWritable, Text, Text, IntWritable>{
        private final static IntWritable one = new IntWritable(1);
        @Override
        public void map(NullWritable key, Text value, Context context
                        ) throws IOException, InterruptedException {
            context.write(value, one);
        }
    }

    public static class IntSumReducer
       extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                        Context context
                        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key,result);
        }
    }

    public static class SortReducer 
    extends Reducer<IntWritable,Text,IntWritable,Text> {
        private int count=1;

        public void reduce(IntWritable key, Iterable<Text> values,
                        Context context
                        ) throws IOException, InterruptedException {
            for (Text val : values) {
                if (count<=100) {
                    context.write(new IntWritable(count),new Text(val.toString()+", "+key.toString()));
                    count++;
                }
            }
        }
    }

    private static class IntWritableDecreasingComparator extends IntWritable.Comparator {
        public int compare(WritableComparable a, WritableComparable b) {
            return -super.compare(a, b);
        }
        public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
            return -super.compare(b1, s1, l1, b2, s2, l2);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        GenericOptionsParser optionparser = new GenericOptionsParser(conf, args);
        conf = optionparser.getConfiguration();

        //连接后输出的临时目录
        Path jointempDir = new Path("MerchantJoin-temp-" + Integer.toString(
                        new Random().nextInt(Integer.MAX_VALUE)));
        //word count后输出的临时目录
        Path wctempDir = new Path("wordcount-temp-" + Integer.toString(
            new Random().nextInt(Integer.MAX_VALUE)));


        Job joinJob = Job.getInstance(conf, "Reduce side join");
        joinJob.setJarByClass(TopMerchants.class);
        joinJob.setMapperClass(JoinMapper.class);
        joinJob.setMapOutputKeyClass(Text.class);
        joinJob.setMapOutputValueClass(Text.class);
        joinJob.setReducerClass(JoinReducer.class);
        joinJob.setOutputKeyClass(NullWritable.class);
        joinJob.setOutputValueClass(Text.class);
        joinJob.setOutputFormatClass(SequenceFileOutputFormat.class);

        FileInputFormat.addInputPaths(joinJob, conf.get("input_data"));
        //join的结果写到临时目录中，下一个word count任务把临时目录作为输入目录
        FileOutputFormat.setOutputPath(joinJob, jointempDir);


        if (joinJob.waitForCompletion(true)) {
            Job wcJob = Job.getInstance(conf, "word count");
            wcJob.setJarByClass(TopMerchants.class);
            wcJob.setMapperClass(TokenizerMapper.class);
            wcJob.setCombinerClass(IntSumReducer.class);
            wcJob.setReducerClass(IntSumReducer.class);
            wcJob.setOutputKeyClass(Text.class);
            wcJob.setOutputValueClass(IntWritable.class);
            wcJob.setInputFormatClass(SequenceFileInputFormat.class);
            wcJob.setOutputFormatClass(SequenceFileOutputFormat.class);

            FileInputFormat.addInputPath(wcJob, jointempDir);
            //word count的结果写到临时目录中，下一个排序任务把临时目录作为输入目录
            FileOutputFormat.setOutputPath(wcJob, wctempDir);

            if (wcJob.waitForCompletion(true)){
                conf.set("mapreduce.output.textoutputformat.separator", ": ");
                Job sortJob = Job.getInstance(conf,"sort");
                //conf.set("mapreduce.output.textoutputformat.separator", ":");
                sortJob.setJarByClass(TopMerchants.class);
                sortJob.setInputFormatClass(SequenceFileInputFormat.class);
                sortJob.setMapperClass(InverseMapper.class);
                sortJob.setReducerClass(SortReducer.class);
                sortJob.setOutputKeyClass(IntWritable.class);
                sortJob.setOutputValueClass(Text.class);
                sortJob.setSortComparatorClass(IntWritableDecreasingComparator.class);
                sortJob.setNumReduceTasks(1);
    
                // 判断输出路径是否存在，如果存在，则删除
                /*
                Path output_dir = new Path(conf.get("output_dir"));
                FileSystem hdfs = output_dir.getFileSystem(conf);
                if (hdfs.isDirectory(output_dir)) {
                    hdfs.delete(output_dir, true);
                }
                */
                FileInputFormat.addInputPath(sortJob, wctempDir);
                FileOutputFormat.setOutputPath(sortJob, new Path(conf.get("output_dir")));
        
                System.exit(sortJob.waitForCompletion(true) ? 0 : 1);
        
            } else {
                System.exit(1);
            }

        } else {
            System.exit(1);
        }
    
    }

}