package topcount;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.InverseMapper;

public class TopItems {

    public static class TokenizerMapper
    extends Mapper<Object, Text, Text, IntWritable>{

        static enum CountersEnum { INPUT_WORDS }

        private final static IntWritable one = new IntWritable(1);

        @Override
        public void map(Object key, Text value, Context context
                        ) throws IOException, InterruptedException {
            String line = value.toString();
            if (line == null || line.equals("")) return;
            String[] words = line.split(",");
            String item_id=words[1];
            String timestamp=words[5];
            String action_type=words[6];
            if (timestamp.equals("1111") && !action_type.equals("0")) {
                context.write(new Text(item_id), one);
            }
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

        Job job = Job.getInstance(conf, "items count");
        job.setJarByClass(TopItems.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
    
        FileInputFormat.addInputPath(job, new Path(args[0]));
    
        Path tempDir = new Path("wordcount-temp-" + Integer.toString(
                        new Random().nextInt(Integer.MAX_VALUE))); //定义一个临时目录
        FileOutputFormat.setOutputPath(job, tempDir);  //将第一个job的结果写到临时目录中，下一个排序任务把临时目录作为输入目录
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
        if (job.waitForCompletion(true)){
            conf.set("mapreduce.output.textoutputformat.separator", ": ");
            Job sortJob = Job.getInstance(conf,"sort");
            //conf.set("mapreduce.output.textoutputformat.separator", ":");
            sortJob.setJarByClass(TopItems.class);
            FileInputFormat.addInputPath(sortJob, tempDir);
            sortJob.setInputFormatClass(SequenceFileInputFormat.class);
    
            sortJob.setMapperClass(InverseMapper.class);
            sortJob.setReducerClass(SortReducer.class);
            sortJob.setNumReduceTasks(1);
            FileOutputFormat.setOutputPath(sortJob, new Path(args[1]));
    
            sortJob.setOutputKeyClass(IntWritable.class);
            sortJob.setOutputValueClass(Text.class);
    
            sortJob.setSortComparatorClass(IntWritableDecreasingComparator.class);
    
            System.exit(sortJob.waitForCompletion(true) ? 0 : 1);
    
        } else {
            System.exit(1);
        }
    }

}