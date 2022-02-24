import torch
import numpy as np
import pickle    

class BatchGenerator(object):
    def __init__(self, actions_dict, gt_path,vid_list_file,vid_list_file_tst,array_dir,feat,SOS_index,EOS_index):
        self.list_of_examples = list()
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.vid_list_file= vid_list_file
        self.vid_list_file_tst= vid_list_file_tst
        self.arrays = array_dir
        self.feat=feat
        self.SOS_index= SOS_index
        self.EOS_index= 0
        
    def get_label_length_seq(self,content,count):
        label_seq = []
        length_seq = []
        start = 0
        for i in range(len(content)):
            if content[i] != content[start]:
                label_seq.append(content[start])
                length_seq.append((i-start)/count)
                start = i
        label_seq.append(content[start])
        length_seq.append((len(content)-start)/count)
    
        return label_seq, length_seq
    
    def splitter(self,gtarray,predarray,split):
        inp=[]
        target=[]
        if split == 2050:
            length=len(gtarray)
            twentyper=(20/100) * length
            nxt50per=(70/100) * length
            inp=predarray[0:int(twentyper)]
            target=gtarray[int(twentyper):int(nxt50per)]
        if split == 2010:
            length=len(gtarray)
            twentyper=(20/100) * length
            nxt10per=(30/100) * length
            inp=predarray[0:int(twentyper)]
            target=gtarray[int(twentyper):int(nxt10per)]  
        if split == 2020:
            length=len(gtarray)
            twentyper=(20/100) * length
            nxt20per=(40/100) * length
            inp=predarray[0:int(twentyper)]
            target=gtarray[int(twentyper):int(nxt20per)]  
        if split == 2030:
            length=len(gtarray)
            twentyper=(20/100) * length
            nxt30per=(50/100) * length
            inp=predarray[0:int(twentyper)]
            target=gtarray[int(twentyper):int(nxt30per)]  
        if split == 3050:
            length=len(gtarray)
            thirtyper=(30/100) * length
            nxt50per=(80/100) * length
            inp=predarray[0:int(thirtyper)]
            target=gtarray[int(thirtyper):int(nxt50per)]
        if split == 3010:
            length=len(gtarray)
            thirtyper=(30/100) * length
            nxt10per=(40/100) * length
            inp=predarray[0:int(thirtyper)]
            target=gtarray[int(thirtyper):int(nxt10per)]  
        if split == 3020:
            length=len(gtarray)
            thirtyper=(30/100) * length
            nxt20per=(50/100) * length
            inp=predarray[0:int(thirtyper)]
            target=gtarray[int(thirtyper):int(nxt20per)]  
        if split == 3030:
            length=len(gtarray)
            thirtyper=(30/100) * length
            nxt30per=(60/100) * length
            inp=predarray[0:int(thirtyper)]
            target=gtarray[int(thirtyper):int(nxt30per)]  
        
        return inp,target
    
    
    def GT_train_data(self):
        input20 =[]
        tar2010 =[]
        tar2020 =[]
        tar2030 =[]
        tar2050 =[]
        
        input30 =[]
        tar3010 =[]
        tar3020 =[]
        tar3030 =[]
        tar3050 =[]

        file_ptr = open(self.vid_list_file, 'r')
        list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        for vid in list_of_examples:
            gt_ptr = open(self.gt_path + vid, 'r')
            content=gt_ptr.read().split('\n')[:-1]
            labels= [item for item in content]
            indexes= [self.actions_dict[item] for item in labels]
            total_frames=len(indexes)
    
            twenty = round(0.2*total_frames)
            thirty = round(0.3*total_frames)
            forty = round(0.4*total_frames)
            fifty = round(0.5*total_frames)
            sixty = round(0.6*total_frames)
            seventy = round(0.7*total_frames)
            eighty = round(0.8*total_frames)
       
            in20 = indexes[:twenty]
            in2010 = indexes[twenty:thirty]
            in2020 = indexes[twenty:forty]
            in2030 = indexes[twenty:fifty]
            in2050 = indexes[twenty:seventy]
    
            in30 = indexes[:thirty]
            in3010 = indexes[thirty:forty]
            in3020 = indexes[thirty:fifty]
            in3030 = indexes[thirty:sixty]
            in3050 = indexes[thirty:eighty]
        
            
            a,b= self.get_label_length_seq(in20,total_frames)
            c,d= self.get_label_length_seq(in2010,total_frames)
            e,f= self.get_label_length_seq(in2020,total_frames)
            g,h= self.get_label_length_seq(in2030,total_frames)
            i,j= self.get_label_length_seq(in2050,total_frames)
    
            k,l= self.get_label_length_seq(in30,total_frames)
            m,n= self.get_label_length_seq(in3010,total_frames)
            o,p= self.get_label_length_seq(in3020,total_frames)
            q,r= self.get_label_length_seq(in3030,total_frames)
            s,t= self.get_label_length_seq(in3050,total_frames)
            
         
    
            twt=[[x,y] for x,y in zip(a,b)]
            input20.append(torch.tensor(twt))
            twt10 =[[x,y] for x,y in zip(c,d)]
            twt10.insert(0,[self.SOS_index,0])
            twt10.append([self.EOS_index,0])
            tar2010.append(torch.tensor(twt10))
            twt20 =[[x,y] for x,y in zip(e,f)]
            twt20.insert(0,[self.SOS_index,0])
            twt20.append([self.EOS_index,0])
            tar2020.append(torch.tensor(twt20))
            twt30 =[[x,y] for x,y in zip(g,h)]
            twt30.insert(0,[self.SOS_index,0])
            twt30.append([self.EOS_index,0])
            tar2030.append(torch.tensor(twt30))
            twt50 =[[x,y] for x,y in zip(i,j)]
            twt50.insert(0,[self.SOS_index,0])
            twt50.append([self.EOS_index,0])
            tar2050.append(torch.tensor(twt50))
    
            trt=[[x,y] for x,y in zip(k,l)]
            input30.append(torch.tensor(trt))
            trt10 =[[x,y] for x,y in zip(m,n)]
            trt10.insert(0,[self.SOS_index,0])
            trt10.append([self.EOS_index,0])
            tar3010.append(torch.tensor(trt10))
            trt20 =[[x,y] for x,y in zip(o,p)]
            trt20.insert(0,[self.SOS_index,0])
            trt20.append([self.EOS_index,0])
            tar3020.append(torch.tensor(trt20))
            trt30 =[[x,y] for x,y in zip(q,r)]
            trt30.insert(0,[self.SOS_index,0])
            trt30.append([self.EOS_index,0])
            tar3030.append(torch.tensor(trt30))
            trt50 =[[x,y] for x,y in zip(s,t)]
            trt50.insert(0,[self.SOS_index,0])
            trt50.append([self.EOS_index,0])
            tar3050.append(torch.tensor(trt50))
            
            
        return input20,tar2010,tar2020,tar2030,tar2050,input30,tar3010,tar3020,tar3030,tar3050
        
    def GT_test_data(self):
        input20_tst =[]
        tar2010_tst =[]
        tar2020_tst =[]
        tar2030_tst =[]
        tar2050_tst =[]
        input30_tst =[]
        tar3010_tst =[]
        tar3020_tst =[]
        tar3030_tst =[]
        tar3050_tst =[]
        file_ptr_tst = open(self.vid_list_file_tst, 'r')
        list_of_examples_tst = file_ptr_tst.read().split('\n')[:-1]
        file_ptr_tst.close()

        for vid in list_of_examples_tst:
            gt_ptr = open(self.gt_path + vid, 'r')
            content=gt_ptr.read().split('\n')[:-1]
            labels= [item for item in content]
            indexes= [self.actions_dict[item] for item in labels]
            total_frames=len(indexes)
    
            twenty = round(0.2*total_frames)
            thirty = round(0.3*total_frames)
            forty = round(0.4*total_frames)
            fifty = round(0.5*total_frames)
            sixty = round(0.6*total_frames)
            seventy = round(0.7*total_frames)
            eighty = round(0.8*total_frames)
    
            in20 = indexes[:twenty]
            in2010 = indexes[twenty:thirty]
            in2020 = indexes[twenty:forty]
            in2030 = indexes[twenty:fifty]
            in2050 = indexes[twenty:seventy]
    
            in30 = indexes[:thirty]
            in3010 = indexes[thirty:forty]
            in3020 = indexes[thirty:fifty]
            in3030 = indexes[thirty:sixty]
            in3050 = indexes[thirty:eighty]
    
            a,b= self.get_label_length_seq(in20,total_frames)
            c,d= self.get_label_length_seq(in2010,total_frames)
            e,f= self.get_label_length_seq(in2020,total_frames)
            g,h= self.get_label_length_seq(in2030,total_frames)
            i,j= self.get_label_length_seq(in2050,total_frames)
    
            k,l= self.get_label_length_seq(in30,total_frames)
            m,n= self.get_label_length_seq(in3010,total_frames)
            o,p= self.get_label_length_seq(in3020,total_frames)
            q,r= self.get_label_length_seq(in3030,total_frames)
            s,t= self.get_label_length_seq(in3050,total_frames)
    
            twt=[[x,y] for x,y in zip(a,b)]
            input20_tst.append(torch.tensor(twt))
    
            twt10 =[[x,y] for x,y in zip(c,d)]
            twt10.insert(0,[self.SOS_index,0])
            twt10.append([self.EOS_index,0])
            tar2010_tst.append(torch.tensor(twt10))
    
            twt20 =[[x,y] for x,y in zip(e,f)]
            twt20.insert(0,[self.SOS_index,0])
            twt20.append([self.EOS_index,0])
            tar2020_tst.append(torch.tensor(twt20))
    
            twt30 =[[x,y] for x,y in zip(g,h)]
            twt30.insert(0,[self.SOS_index,0])
            twt30.append([self.EOS_index,0])
            tar2030_tst.append(torch.tensor(twt30))
    
            twt50 =[[x,y] for x,y in zip(i,j)]
            twt50.insert(0,[self.SOS_index,0])
            twt50.append([self.EOS_index,0])
            tar2050_tst.append(torch.tensor(twt50))
    
            trt=[[x,y] for x,y in zip(k,l)]
            input30_tst.append(torch.tensor(trt))
    
            trt10 =[[x,y] for x,y in zip(m,n)]
            trt10.insert(0,[self.SOS_index,0])
            trt10.append([self.EOS_index,0])
            tar3010_tst.append(torch.tensor(trt10))
            
            trt20 =[[x,y] for x,y in zip(o,p)]
            trt20.insert(0,[self.SOS_index,0])
            trt20.append([self.EOS_index,0])
            tar3020_tst.append(torch.tensor(trt20))
            
            trt30 =[[x,y] for x,y in zip(q,r)]
            trt30.insert(0,[self.SOS_index,0])
            trt30.append([self.EOS_index,0])
            tar3030_tst.append(torch.tensor(trt30))
            
            trt50 =[[x,y] for x,y in zip(s,t)]
            trt50.insert(0,[self.SOS_index,0])
            trt50.append([self.EOS_index,0])
            tar3050_tst.append(torch.tensor(trt50))
        return input20_tst,tar2010_tst,tar2020_tst,tar2030_tst,tar2050_tst,input30_tst,tar3010_tst,tar3020_tst,tar3030_tst,tar3050_tst
    
    
    def GT_train_data_feat(self):
        input20 =[]
        tar2010 =[]
        tar2020 =[]
        tar2030 =[]
        tar2050 =[]
        
        input30 =[]
        tar3010 =[]
        tar3020 =[]
        tar3030 =[]
        tar3050 =[]

        file_ptr = open(self.vid_list_file, 'r')
        list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        for vid in list_of_examples:
            gt_ptr = open(self.gt_path + vid, 'r')
            feat_ptr=np.load(self.feat+vid.replace(".txt",".npy"))
            
            content=gt_ptr.read().split('\n')[:-1]
            labels= [item for item in content]
            indexes= [self.actions_dict[item] for item in labels]
            total_frames=len(indexes)
        
            twenty = round(0.2*total_frames)
            thirty = round(0.3*total_frames)
            forty = round(0.4*total_frames)
            fifty = round(0.5*total_frames)
            sixty = round(0.6*total_frames)
            seventy = round(0.7*total_frames)
            eighty = round(0.8*total_frames)
           
            in20 = indexes[:twenty]
            in2010 = indexes[twenty:thirty]
            in2020 = indexes[twenty:forty]
            in2030 = indexes[twenty:fifty]
            in2050 = indexes[twenty:seventy]
    
            in30 = indexes[:thirty]
            in3010 = indexes[thirty:forty]
            in3020 = indexes[thirty:fifty]
            in3030 = indexes[thirty:sixty]
            in3050 = indexes[thirty:eighty]
            
            
            #a,b= self.get_label_length_seq(in20,total_frames)
            c,d= self.get_label_length_seq(in2010,total_frames)
            e,f= self.get_label_length_seq(in2020,total_frames)
            g,h= self.get_label_length_seq(in2030,total_frames)
            i,j= self.get_label_length_seq(in2050,total_frames)
    
            #k,l= self.get_label_length_seq(in30,total_frames)
            m,n= self.get_label_length_seq(in3010,total_frames)
            o,p= self.get_label_length_seq(in3020,total_frames)
            q,r= self.get_label_length_seq(in3030,total_frames)
            s,t= self.get_label_length_seq(in3050,total_frames)
            
         
    
            #twt=[[x,y] for x,y in zip(a,b)]
            #input20.append(torch.tensor(twt))
            twt=np.transpose(feat_ptr[:,:twenty:5])
            input20.append(torch.tensor(twt))
            
            twt10 =[[x,y] for x,y in zip(c,d)]
            twt10.insert(0,[self.SOS_index,0])
            twt10.append([self.EOS_index,0])
            tar2010.append(torch.tensor(twt10))
            twt20 =[[x,y] for x,y in zip(e,f)]
            twt20.insert(0,[self.SOS_index,0])
            twt20.append([self.EOS_index,0])
            tar2020.append(torch.tensor(twt20))
            twt30 =[[x,y] for x,y in zip(g,h)]
            twt30.insert(0,[self.SOS_index,0])
            twt30.append([self.EOS_index,0])
            tar2030.append(torch.tensor(twt30))
            twt50 =[[x,y] for x,y in zip(i,j)]
            twt50.insert(0,[self.SOS_index,0])
            twt50.append([self.EOS_index,0])
            tar2050.append(torch.tensor(twt50))
    
            #trt=[[x,y] for x,y in zip(k,l)]
            #input30.append(torch.tensor(trt))
            trt=np.transpose(feat_ptr[:,:thirty:5])
            input30.append(torch.tensor(trt))
            
            trt10 =[[x,y] for x,y in zip(m,n)]
            trt10.insert(0,[self.SOS_index,0])
            trt10.append([self.EOS_index,0])
            tar3010.append(torch.tensor(trt10))
            trt20 =[[x,y] for x,y in zip(o,p)]
            trt20.insert(0,[self.SOS_index,0])
            trt20.append([self.EOS_index,0])
            tar3020.append(torch.tensor(trt20))
            trt30 =[[x,y] for x,y in zip(q,r)]
            trt30.insert(0,[self.SOS_index,0])
            trt30.append([self.EOS_index,0])
            tar3030.append(torch.tensor(trt30))
            trt50 =[[x,y] for x,y in zip(s,t)]
            trt50.insert(0,[self.SOS_index,0])
            trt50.append([self.EOS_index,0])
            tar3050.append(torch.tensor(trt50))
            
            
        return input20,tar2010,tar2020,tar2030,tar2050,input30,tar3010,tar3020,tar3030,tar3050
    
    def GT_test_data_feat(self):
        input20 =[]
        tar2010 =[]
        tar2020 =[]
        tar2030 =[]
        tar2050 =[]
        
        input30 =[]
        tar3010 =[]
        tar3020 =[]
        tar3030 =[]
        tar3050 =[]

        file_ptr_tst = open(self.vid_list_file_tst, 'r')
        list_of_examples_tst = file_ptr_tst.read().split('\n')[:-1]
        file_ptr_tst.close()

        for vid in list_of_examples_tst:
            gt_ptr = open(self.gt_path + vid, 'r')
            feat_ptr=np.load(self.feat+vid.replace(".txt",".npy"))
            
            content=gt_ptr.read().split('\n')[:-1]
            labels= [item for item in content]
            indexes= [self.actions_dict[item] for item in labels]
            total_frames=len(indexes)
        
            twenty = round(0.2*total_frames)
            thirty = round(0.3*total_frames)
            forty = round(0.4*total_frames)
            fifty = round(0.5*total_frames)
            sixty = round(0.6*total_frames)
            seventy = round(0.7*total_frames)
            eighty = round(0.8*total_frames)
           
            in20 = indexes[:twenty]
            in2010 = indexes[twenty:thirty]
            in2020 = indexes[twenty:forty]
            in2030 = indexes[twenty:fifty]
            in2050 = indexes[twenty:seventy]
    
            in30 = indexes[:thirty]
            in3010 = indexes[thirty:forty]
            in3020 = indexes[thirty:fifty]
            in3030 = indexes[thirty:sixty]
            in3050 = indexes[thirty:eighty]
            
            
            #a,b= self.get_label_length_seq(in20,total_frames)
            c,d= self.get_label_length_seq(in2010,total_frames)
            e,f= self.get_label_length_seq(in2020,total_frames)
            g,h= self.get_label_length_seq(in2030,total_frames)
            i,j= self.get_label_length_seq(in2050,total_frames)
    
            #k,l= self.get_label_length_seq(in30,total_frames)
            m,n= self.get_label_length_seq(in3010,total_frames)
            o,p= self.get_label_length_seq(in3020,total_frames)
            q,r= self.get_label_length_seq(in3030,total_frames)
            s,t= self.get_label_length_seq(in3050,total_frames)
            
            #twt=[[x,y] for x,y in zip(a,b)]
            #input20.append(torch.tensor(twt))
            twt=np.transpose(feat_ptr[:,:twenty:5])
            input20.append(torch.tensor(twt))
            
            twt10 =[[x,y] for x,y in zip(c,d)]
            twt10.insert(0,[self.SOS_index,0])
            twt10.append([self.EOS_index,0])
            tar2010.append(torch.tensor(twt10))
            twt20 =[[x,y] for x,y in zip(e,f)]
            twt20.insert(0,[self.SOS_index,0])
            twt20.append([self.EOS_index,0])
            tar2020.append(torch.tensor(twt20))
            twt30 =[[x,y] for x,y in zip(g,h)]
            twt30.insert(0,[self.SOS_index,0])
            twt30.append([self.EOS_index,0])
            tar2030.append(torch.tensor(twt30))
            twt50 =[[x,y] for x,y in zip(i,j)]
            twt50.insert(0,[self.SOS_index,0])
            twt50.append([self.EOS_index,0])
            tar2050.append(torch.tensor(twt50))
    
            #trt=[[x,y] for x,y in zip(k,l)]
            #input30.append(torch.tensor(trt))
            trt=np.transpose(feat_ptr[:,:thirty:5])
            input30.append(torch.tensor(trt))
            
            trt10 =[[x,y] for x,y in zip(m,n)]
            trt10.insert(0,[self.SOS_index,0])
            trt10.append([self.EOS_index,0])
            tar3010.append(torch.tensor(trt10))
            trt20 =[[x,y] for x,y in zip(o,p)]
            trt20.insert(0,[self.SOS_index,0])
            trt20.append([self.EOS_index,0])
            tar3020.append(torch.tensor(trt20))
            trt30 =[[x,y] for x,y in zip(q,r)]
            trt30.insert(0,[self.SOS_index,0])
            trt30.append([self.EOS_index,0])
            tar3030.append(torch.tensor(trt30))
            trt50 =[[x,y] for x,y in zip(s,t)]
            trt50.insert(0,[self.SOS_index,0])
            trt50.append([self.EOS_index,0])
            tar3050.append(torch.tensor(trt50))
            
            
        return input20,tar2010,tar2020,tar2030,tar2050,input30,tar3010,tar3020,tar3030,tar3050
    
    
    def ASRF_pred(self):
        file_ptr_tst = open(self.vid_list_file_tst, 'r')
        list_of_examples_tst = file_ptr_tst.read().split('\n')[:-1]
        file_ptr_tst.close()
        
        input20_pred=[]
        tar2010_pred =[]
        tar2020_pred =[]
        tar2030_pred =[]
        tar2050_pred =[]
        input30_pred =[]
        tar3010_pred =[]
        tar3020_pred =[]
        tar3030_pred =[]
        tar3050_pred =[]
        
        for vid in list_of_examples_tst:
            vid = vid[:-4]
            gt_array = np.load(self.arrays + vid + "_gt.npy")
            gt_array = [x+1 for x in gt_array]
            pred_array = np.load(self.arrays + vid + "_refined_pred.npy")
            pred_array=[x+1 for x in pred_array]
            count = 0
            
            for item in gt_array:
                count+= 1
            prediction20,gt_tar2010= self.splitter(gt_array, pred_array, 2010)
            prediction20,gt_tar2020= self.splitter(gt_array, pred_array, 2020)
            prediction20,gt_tar2030= self.splitter(gt_array, pred_array, 2030)
            prediction20,gt_tar2050= self.splitter(gt_array, pred_array, 2050)
            
            prediction30,gt_tar3010= self.splitter(gt_array, pred_array, 3010)
            prediction30,gt_tar3020= self.splitter(gt_array, pred_array, 3020)
            prediction30,gt_tar3030= self.splitter(gt_array, pred_array, 3030)
            prediction30,gt_tar3050= self.splitter(gt_array, pred_array, 3050)
            
            a,b= self.get_label_length_seq(prediction20,count)
            c,d= self.get_label_length_seq(gt_tar2010,count)
            e,f= self.get_label_length_seq(gt_tar2020,count)
            g,h= self.get_label_length_seq(gt_tar2030,count)
            k,l= self.get_label_length_seq(gt_tar2050,count)
            
            m,n= self.get_label_length_seq(prediction30,count)
            o,p= self.get_label_length_seq(gt_tar3010,count)
            q,r= self.get_label_length_seq(gt_tar3020,count)
            s,t= self.get_label_length_seq(gt_tar3030,count)
            u,v= self.get_label_length_seq(gt_tar3050,count)
            
            
            twt=[[a[i],b[i]]for i in range(len(a))]
            input20_pred.append(torch.tensor(twt))
    
            twt10 =[[c[i],d[i]]for i in range(len(c))]
            twt10.insert(0,[self.SOS_index,0])
            twt10.append([self.EOS_index,0])
            tar2010_pred.append(torch.tensor(twt10))
    
            twt20 =[[e[i],f[i]]for i in range(len(e))]
            twt20.insert(0,[self.SOS_index,0])
            twt20.append([self.EOS_index,0])
            tar2020_pred.append(torch.tensor(twt20))
    
            twt30 =[[g[i],h[i]]for i in range(len(g))]
            twt30.insert(0,[self.SOS_index,0])
            twt30.append([self.EOS_index,0])
            tar2030_pred.append(torch.tensor(twt30))
    
            twt50 =[[k[i],l[i]]for i in range(len(k))]
            twt50.insert(0,[self.SOS_index,0])
            twt50.append([self.EOS_index,0])
            tar2050_pred.append(torch.tensor(twt50))
    
            trt=[[m[i],n[i]]for i in range(len(m))]
            input30_pred.append(torch.tensor(trt))
    
            trt10 =[[o[i],p[i]]for i in range(len(o))]
            trt10.insert(0,[self.SOS_index,0])
            trt10.append([self.EOS_index,0])
            tar3010_pred.append(torch.tensor(trt10))
            
            trt20 =[[q[i],r[i]]for i in range(len(q))]
            trt20.insert(0,[self.SOS_index,0])
            trt20.append([self.EOS_index,0])
            tar3020_pred.append(torch.tensor(trt20))
            
            trt30 =[[s[i],t[i]]for i in range(len(s))]
            trt30.insert(0,[self.SOS_index,0])
            trt30.append([self.EOS_index,0])
            tar3030_pred.append(torch.tensor(trt30))
            
            trt50 =[[u[i],v[i]]for i in range(len(u))]
            trt50.insert(0,[self.SOS_index,0])
            trt50.append([self.EOS_index,0])
            tar3050_pred.append(torch.tensor(trt50))
            
            
            
        return input20_pred,tar2010_pred,tar2020_pred,tar2030_pred,tar2050_pred,input30_pred,tar3010_pred,tar3020_pred,tar3030_pred,tar3050_pred
    
class LabelsDataset(torch.utils.data.Dataset):
    def __init__(self, input,targets):
        self.data_input = input
        self.data_targets = targets
        
    # get sample
    def __getitem__(self, idx):
        inp = self.data_input[idx]
        tar = self.data_targets[idx]
        return inp, tar
    
    def __len__(self):
        return len(self.data_input)

def randcuts(dataset):
    if dataset == "50Salads":
        with open("list_of_examples.txt", "rb") as fp:   #Pickling
            examps=pickle.load(fp)
    if dataset == "breakfast" :  
        with open("breakfast.txt", "rb") as fp:   #Pickling
            examps=pickle.load(fp)
    seqs=[]
    tars=[]    
    for item in examps:
        seqs.append(torch.tensor(item[0]).float())
        tars.append(torch.tensor(item[1]).float())
    return seqs,tars


    


    