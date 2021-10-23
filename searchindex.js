Search.setIndex({docnames:["components/attentions","components/feedforward","components/index","components/mha","components/position_embedding","components/reversible","custom_parts/index","factory/block","factory/index","factory/model","index","tools/index","tutorials/blocksparse","tutorials/extend_attentions","tutorials/index","tutorials/pytorch_encoder","tutorials/reversible","tutorials/sparse_vit","tutorials/triton","tutorials/use_attention","what_is_xformers"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["components/attentions.rst","components/feedforward.rst","components/index.rst","components/mha.rst","components/position_embedding.rst","components/reversible.rst","custom_parts/index.rst","factory/block.rst","factory/index.rst","factory/model.rst","index.rst","tools/index.rst","tutorials/blocksparse.rst","tutorials/extend_attentions.rst","tutorials/index.rst","tutorials/pytorch_encoder.rst","tutorials/reversible.rst","tutorials/sparse_vit.rst","tutorials/triton.rst","tutorials/use_attention.rst","what_is_xformers.rst"],objects:{"xformers.components":{MultiHeadDispatch:[3,0,1,""],attention:[0,3,0,"-"],feedforward:[1,3,0,"-"],positional_embedding:[4,3,0,"-"]},"xformers.components.MultiHeadDispatch":{forward:[3,1,1,""],from_config:[3,1,1,""],training:[3,2,1,""]},"xformers.components.attention":{Attention:[0,0,1,""],FavorAttention:[0,0,1,""],GlobalAttention:[0,0,1,""],LinformerAttention:[0,0,1,""],LocalAttention:[0,0,1,""],NystromAttention:[0,0,1,""],OrthoFormerAttention:[0,0,1,""],RandomAttention:[0,0,1,""],ScaledDotProduct:[0,0,1,""],build_attention:[0,4,1,""],register_attention:[0,4,1,""]},"xformers.components.attention.Attention":{forward:[0,1,1,""],from_config:[0,1,1,""]},"xformers.components.attention.FavorAttention":{__init__:[0,1,1,""],forward:[0,1,1,""]},"xformers.components.attention.GlobalAttention":{__init__:[0,1,1,""],forward:[0,1,1,""]},"xformers.components.attention.LinformerAttention":{__init__:[0,1,1,""],forward:[0,1,1,""]},"xformers.components.attention.LocalAttention":{__init__:[0,1,1,""],forward:[0,1,1,""]},"xformers.components.attention.NystromAttention":{__init__:[0,1,1,""],forward:[0,1,1,""]},"xformers.components.attention.OrthoFormerAttention":{__init__:[0,1,1,""],forward:[0,1,1,""]},"xformers.components.attention.RandomAttention":{__init__:[0,1,1,""],forward:[0,1,1,""]},"xformers.components.attention.ScaledDotProduct":{forward:[0,1,1,""],mask:[0,2,1,""]},"xformers.components.feedforward":{Feedforward:[1,0,1,""],MLP:[1,0,1,""],build_feedforward:[1,4,1,""],register_feedforward:[1,4,1,""]},"xformers.components.feedforward.Feedforward":{from_config:[1,1,1,""],training:[1,2,1,""]},"xformers.components.feedforward.MLP":{forward:[1,1,1,""],training:[1,2,1,""]},"xformers.components.positional_embedding":{SinePositionalEmbedding:[4,0,1,""],VocabEmbedding:[4,0,1,""],build_positional_embedding:[4,4,1,""],register_positional_embedding:[4,4,1,""]},"xformers.components.positional_embedding.SinePositionalEmbedding":{forward:[4,1,1,""],training:[4,2,1,""]},"xformers.components.positional_embedding.VocabEmbedding":{forward:[4,1,1,""],training:[4,2,1,""]},"xformers.components.reversible":{Deterministic:[5,0,1,""],ReversibleBlock:[5,0,1,""],ReversibleSequence:[5,0,1,""]},"xformers.components.reversible.Deterministic":{forward:[5,1,1,""],record_rng:[5,1,1,""],training:[5,2,1,""]},"xformers.components.reversible.ReversibleBlock":{backward_pass:[5,1,1,""],forward:[5,1,1,""],training:[5,2,1,""]},"xformers.components.reversible.ReversibleSequence":{forward:[5,1,1,""],training:[5,2,1,""]},"xformers.factory":{block_factory:[7,3,0,"-"],model_factory:[9,3,0,"-"]},"xformers.factory.block_factory":{BlockType:[7,0,1,""],LayerPosition:[7,0,1,""],LayerPositionBitmask:[7,0,1,""],xFormerBlockConfig:[7,0,1,""],xFormerDecoderBlock:[7,0,1,""],xFormerDecoderConfig:[7,0,1,""],xFormerEncoderBlock:[7,0,1,""],xFormerEncoderConfig:[7,0,1,""]},"xformers.factory.block_factory.BlockType":{Decoder:[7,2,1,""],Encoder:[7,2,1,""]},"xformers.factory.block_factory.LayerPosition":{is_first:[7,1,1,""],is_last:[7,1,1,""],mark_not_first:[7,1,1,""],mark_not_last:[7,1,1,""]},"xformers.factory.block_factory.LayerPositionBitmask":{Default:[7,2,1,""],First:[7,2,1,""],Last:[7,2,1,""]},"xformers.factory.block_factory.xFormerBlockConfig":{block_type:[7,2,1,""],dim_model:[7,2,1,""],feedforward_config:[7,2,1,""],layer_norm_style:[7,2,1,""],layer_position:[7,2,1,""],position_encoding_config:[7,2,1,""],use_triton:[7,2,1,""]},"xformers.factory.block_factory.xFormerDecoderBlock":{forward:[7,1,1,""],from_config:[7,1,1,""],training:[7,2,1,""]},"xformers.factory.block_factory.xFormerDecoderConfig":{multi_head_config_cross:[7,2,1,""],multi_head_config_masked:[7,2,1,""]},"xformers.factory.block_factory.xFormerEncoderBlock":{forward:[7,1,1,""],from_config:[7,1,1,""],get_reversible_layer:[7,1,1,""],training:[7,2,1,""]},"xformers.factory.block_factory.xFormerEncoderConfig":{multi_head_config:[7,2,1,""],use_triton:[7,2,1,""]},"xformers.factory.model_factory":{xFormer:[9,0,1,""],xFormerConfig:[9,0,1,""],xFormerStackConfig:[9,0,1,""]},"xformers.factory.model_factory.xFormer":{__init__:[9,1,1,""],forward:[9,1,1,""],from_config:[9,1,1,""],training:[9,2,1,""]},"xformers.factory.model_factory.xFormerConfig":{stack_configs:[9,2,1,""]},"xformers.factory.model_factory.xFormerStackConfig":{block_config:[9,2,1,""],num_layers:[9,2,1,""],reversible:[9,2,1,""]},xformers:{triton:[6,3,0,"-"]}},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"],"2":["py","attribute","Python attribute"],"3":["py","module","Python module"],"4":["py","function","Python function"]},objtypes:{"0":"py:class","1":"py:method","2":"py:attribute","3":"py:module","4":"py:function"},terms:{"1024":[12,15,18,19],"128":18,"133":18,"136":18,"149":18,"151mb":12,"153":18,"164":18,"168":18,"170":18,"192":18,"193":18,"196":18,"1e3":12,"1e6":12,"2017":16,"2020":[0,16],"2021":[0,18],"203":18,"204":18,"2048":[12,15,18],"205":18,"206":18,"207":18,"211":18,"213":18,"215":18,"218":18,"220":18,"224":[17,18],"229":18,"234":18,"247":18,"251":18,"252":18,"254":18,"255":18,"256":18,"257":18,"258":18,"263":18,"264":18,"265":18,"266":18,"271":18,"273":18,"284":18,"285":18,"289":18,"290":18,"291":18,"294":18,"295":18,"301":18,"307":18,"3136":18,"341":18,"384":[15,18,19],"393mb":12,"399":18,"4096":18,"496":18,"501":18,"512":[15,18],"522":18,"524":18,"534":18,"537":18,"541":18,"543":18,"545":18,"553":18,"555":18,"558":18,"570":18,"597":18,"601":18,"606":18,"615":18,"619m":12,"636":18,"650":18,"660":18,"669":18,"678":18,"682":18,"696":18,"702":18,"712":18,"716":18,"736":18,"744":18,"748":18,"760":18,"775":18,"777":18,"780":18,"784":18,"789":18,"796":18,"799":18,"809":18,"814":18,"837m":12,"\u0142":16,"abstract":0,"boolean":16,"break":20,"case":[0,17,18,19],"class":[0,1,3,4,5,7,9,13,16],"default":[6,7,13],"enum":7,"export":6,"float":[0,1,3,4,13],"function":[12,16,17],"import":[12,15,17,18,19],"int":[0,1,3,4,7,9,13,16],"new":[13,19,20],"return":17,"short":18,"static":7,"true":[0,3,5,6,7,12,13,16,18],"try":12,"while":[0,15],And:20,Doing:13,Eye:0,For:[0,1,4],Its:16,One:16,PRs:20,That:19,The:[0,3,6,8,12,13,15,16,17,18,19],Their:15,There:[6,13,15,18,19],These:6,Using:[10,14],__init__:[0,9,13,16],_bigbird:0,_you:12,abl:6,ablat:20,abov:[6,16,17],accept:[0,13],accuraci:16,across:20,activ:[1,6,15,16,18],actual:[3,12,18],adapt:16,add_modul:17,addit:0,affect:16,after:12,against:20,agnost:20,ahm:0,aim:20,ainsli:0,alberti:0,algorithm:0,all:[0,3,10,13,14,16,18,19,20],allow:[0,1,4,19,20],alon:20,along:[12,18],alreadi:[12,15,17],also:[6,13,16,18,20],alter:17,altern:[17,19],although:19,amp:18,ani:[0,1,4,6,7,9,13,17,18,20],anoth:16,anyth:[12,20],anytim:20,api:[10,19],appli:[0,13,16],applic:[13,16,18],approxim:0,architectur:[3,13,19,20],architur:6,arg:[0,1,3,4,5,9,13],arg_rout:[5,16],argument:13,art:[10,20],arxiv:0,asano:0,aspect:17,assum:[0,1,4],att_mask:[0,3,7,12,17],att_val:12,attend:0,attent:[1,2,4,6,10,12,13,14,15,16],attention_mask:0,attention_nam:19,attention_pattern:17,attention_query_mask:[0,19],attentionconfig:[0,13],attn_mask:[12,17],autograd:18,automat:[6,13,20],avail:18,awar:[0,3,17,18],backend:18,backpropag:16,backward_pass:5,ball:0,bar:[0,1,4],base:[0,1,3,4,6,7,9,10,13,14,17,19],batch:[0,3,12,13,19],batch_first:[12,15],batch_submit:13,becom:[16,19],befor:0,being:[0,12,18,20],benchmark:[8,13,15,20],benefit:16,better:[12,18],between:[12,16],bia:[3,18],bias:12,big:0,bigbird:0,binari:12,bird:0,bitmask:7,block:[5,6,8,9,10,12,13,14,15,20],block_config:[9,15,16],block_factori:[7,9],block_siz:12,block_typ:[7,15],blockif:12,blockspars:12,blocksparseattent:[10,14],blocktyp:7,bonu:12,bookeep:12,bool:[0,1,3,4,5,7,9,13,16],both:[0,7,13],bruegger:16,build:[0,1,4,10,12,16,19,20],build_attent:[0,19],build_feedforward:1,build_positional_embed:4,built:19,bypass:9,call:[0,1,4,13],campbel:0,can:[0,3,6,9,10,12,13,15,16,17,18,19,20],cannot:[0,20],capabl:[6,18],care:15,causal:[0,12,13,15],causal_layout:12,causal_mask:12,certain:0,chakraborti:0,chang:[6,12,15],check:[15,17],checkpo:13,checkpoint:16,child:17,chunk:16,classmethod:[0,1,3,7,9],clear:17,clone:6,close:[15,16],closer:17,cluster:13,code:[6,13,18,19],codebas:0,coeffici:12,com:[0,6],combin:[0,10,16,19],commod:12,common:6,commun:16,compar:[12,16,18,20],compat:[16,18],compil:[6,18],complet:[8,16,19],complex:0,compon:[0,1,3,4,5,7,10,12,13,16,17,19],compos:20,comprehens:6,comput:[6,12,16],conda:6,condit:6,config:[0,1,3,4,7,9,13,15],config_path:13,configur:[0,1,4,9,13],consid:[13,16,17],consolid:18,constant_mask:0,constitut:15,construct:[12,13],constructor:13,contain:5,context:16,conv_kernel_s:[0,13],convert:12,correct:0,correspond:[9,18],cost:16,could:[12,15],coupl:[6,13,15,18],creat:[6,10],crowd:20,cuda:[10,12,17,18],current:[6,18],custom:10,custom_decod:15,custom_encod:15,d_model:15,data:[12,19],dataclass:13,ddp:16,deal:16,declar:15,decod:[7,15],decoder_att_mask:7,decoder_input_mask:9,decor:[0,1,4],dedic:13,def:[12,13,16,17],defer:19,defin:[9,12,13,16,17,20],definit:[9,19],del:17,dens:12,depend:19,depth:17,design:20,detail:16,determin:[0,1,4],determinist:5,develop:8,devic:[12,15,19],dict:[0,1,4,7,9,19],differ:[12,13,15,16],dim:[3,16,17],dim_featur:0,dim_feedforward:15,dim_head:0,dim_kei:3,dim_model:[1,3,4,7,12,15,19],dim_valu:3,dimens:[0,3,12,15,18,19],directli:[13,18,19],dispatch:[3,19],distribut:[0,16],doe:16,doesn:15,domain:20,done:13,dot:[0,6],drop:[0,12,18],dropout:[0,1,4,12,13,15,19],dtype:[12,15],dubei:0,dummi:19,each:[0,18,20],easi:20,easili:[9,19,20],effect:16,effici:16,effort:20,either:[18,19],element:0,emb:[3,12],embed:[2,10],embed_dim:17,empti:[0,12],empty_cach:12,enabl:[6,13,18],encod:[4,7,10,14],encoder_att_mask:7,encoder_input_mask:9,encodinhg:15,end:3,engin:20,enough:[6,12],entir:15,enumer:7,env:6,equival:[10,14],even:[0,1,4,13],evenli:0,everi:20,everyth:19,exact:[13,15],exactli:16,exampl:[6,12,17,18],exchang:17,exhaust:17,exhibit:17,exist:[10,14],expect:[0,3,18,19],experi:17,explicitli:13,expos:[13,15,16,17,19],extend:[10,14,20],extens:[13,20],extra:[12,13,16,17],f_arg:[5,16],facebookresearch:0,fact:12,factor:9,factori:[10,15,16],fairintern:6,fairli:15,fals:[0,3,5,9,13,15,16,17,18],famili:18,fast:[0,12,20],faster:18,favor:0,favorattent:0,featur:0,feature_map:0,feature_map_typ:0,featuremaptyp:0,feedforward:[2,7,10,16],feedforward_config:[7,15],feedforwardconfig:[1,7],feichtenhof:0,field:[13,20],file:[0,1,4,19],find:[0,1,4],fine:[12,19],first:[7,16],flag:19,flexibl:[10,15,20],float16:[12,18],float32:18,focu:20,focus:20,follow:[3,6,13,15,16,17,18,20],foo:[0,1,4],force_spars:0,fork:13,formul:16,forward:[0,1,3,4,5,7,9,12,13,16],fp16:18,free:16,from:[0,1,4,6,10,12,13,14,15,16,18,19],from_config:[0,1,3,4,7,9,15],fullfil:6,fung:0,fusedlinearlay:18,futur:0,fututur:15,g_arg:[5,16],gcc:6,gener:9,get:[12,16,17,18],get_reversible_lay:7,git:6,github:[0,6],given:[0,1,4,9,13,17,19,20],global:0,globalattent:0,glr:18,goal:18,gomez:16,good:17,gpu:[6,16,18],grain:12,gross:16,guruganesh:0,half:12,happen:20,has:15,have:[10,12,14,17],head:[0,2,10,12,16,19],head_:12,heavi:20,help:[0,16],helper:[9,12,16,17,19],henriqu:0,here:[0,10,12,14,15],hidden_layer_multipli:[1,15],hoc:18,hopefulli:19,host:[10,14],how:[17,18],html:18,http:[0,18],idea:17,ideal:20,ignor:0,imag:17,img_siz:17,implement:[0,6,7,16],improv:20,in_featur:18,in_proj_contain:3,includ:16,increas:[16,18],independ:6,inform:13,inherit:13,initi:19,inprojcontain:3,input:[1,3,16,19],input_mask:7,inspir:20,instal:6,instanc:[0,1,4,13,15],instanti:[0,1,4,15,19],interest:[10,14],interfac:[13,15,18],intern:19,interoper:[10,20],inv_iter:[0,13],is_first:7,is_last:7,isinst:17,issu:6,iter:0,iter_before_redraw:0,itself:[0,1,4],jit:18,job:13,json:13,just:[12,17,18],kaiser:16,keep:0,kei:[0,1,3,4,12],kept:0,kernel:[0,10,18],kitaev:16,knob:15,kwarg:[0,1,3,4,5,7,9,12,13,16],label:0,landmark_pool:[0,13],landmark_select:0,landmarkselect:0,lang:18,languag:18,larg:16,last:[7,18],layer:[2,7,10,14,15,16],layer_norm_ep:15,layer_norm_styl:[7,15],layer_posit:7,layernorm:17,layernormstyl:7,layerposit:7,layerpositionbitmask:7,layout:12,least:13,length:[0,3,12],less:[6,16],let:[12,13,17],levskaya:16,librari:[0,1,4,10,20],lightli:16,like:[13,15,16,17,18,19],limit:[6,12,18],line:[12,18],linear:0,linform:[0,15],linformerattent:0,list:[9,15],littl:[15,19],load:6,local:[0,13,20],localattent:0,log:[13,18],log_softmax:18,longer:0,longform:0,look:[12,15],loos:13,lot:19,lower:12,lra:[13,15],lsh:16,lucidrain:16,machin:6,made:16,mai:17,main:16,make:[3,12,15,16,18,19],mani:18,manual:17,map:0,mark:7,mark_not_first:7,mark_not_last:7,mask:[0,3,6,12,17,19],match:[6,12],matrix:12,max_memori:12,max_memory_alloc:12,maxpool:12,maybe_merge_mask:0,mean:[0,15,16,20],measur:[18,20],mechan:[2,3,10,13,14,15,17],mem_us:12,memori:[0,7,12,15,16],merg:0,met:6,method:0,metz:0,mha:16,mind:[12,17],minim:17,minimum:12,misra:0,mlp:[1,15,16],mlp_ratio:17,mlpen:0,model:[6,8,10,14,15,16,19,20],model_factori:[9,15,16],modul:[0,1,3,5,6,7,9,13,16,17],module_output:17,modulelist:[5,16],monkei:17,more:[12,13,15,16,17,19,20],moreov:16,mostli:15,motionform:0,move:20,multi:[0,2,10,16,19],multi_head:[12,19],multi_head_config:[7,15],multi_head_config_cross:7,multi_head_config_mask:7,multi_head_dispatch:3,multihead:[12,19],multiheadattent:12,multiheaddispatch:[3,12,19],multiheaddispatchconfig:3,multipl:16,must:0,my_attent:0,my_component_nam:13,my_config:[15,19],my_fancy_mask:17,my_feedforward:1,my_linear_lay:18,my_position_encod:4,name:[0,1,4,6,13,15,17,19],named_children:17,natur:16,need:[0,3,6,12,13,15,16],neg:0,net:5,network:16,neurip:0,nhead:15,non:18,none:[0,1,3,7,9,13,15,18],nonetyp:7,norm:16,norm_lay:17,normalize_input:0,notat:0,note:[7,12,15,16,17],notebook:17,noth:[6,7],now:[12,13,17],num_decoder_lay:15,num_encoder_lay:15,num_head:[0,3,12,13,15,17,19],num_landmark:[0,13],num_lay:[9,15,16],number:[0,16,18],nvcc:6,nvidia:18,nystrom:[0,13],nystromattent:[0,13],nystromform:0,nystromselfattentionconfig:13,object:[7,9,15],obscur:19,odd:0,often:6,one:[12,13,16,17,20],ones:12,onli:[0,6,9,10,14,18],ontanon:0,open:[13,16],oper:18,operand:18,optim:[10,18,20],option:[0,1,3,7,8,9,10,13,15,16],order:13,org:18,origin:[0,16],ortho:0,orthoform:0,orthoformerattent:0,orthogon:0,other:[0,13,16,17,18,20],our:[12,17],out:[10,12,14],out_featur:18,out_proj:3,output:[0,3,16,17],over:18,overal:0,own:19,pad:0,paper:[13,16],parallel:18,paramet:[0,17,19],part:[0,1,4,10,14,15],particular:[12,17],pass:[0,12,16],patch:17,patch_siz:17,path:[13,16,19],patrick:0,pattern:[12,17],peak:12,per:12,perfect:12,perform:[0,18],pham:0,pick:13,pinverse_original_init:[0,13],pip:6,pleas:[12,16,17],point:[13,18],posit:[0,2,10,15],position_encoding_config:[7,15],positional_embed:[4,7],positionembed:4,positionembeddingconfig:[4,7],possibl:[12,18,19],post:[7,15],power:12,practic:[13,17],pre:15,precis:17,present:6,primarli:8,primit:18,print:12,privat:13,probabl:[0,12,15],process:16,produc:16,product:[0,6],program:18,programat:19,progress:13,project:3,propos:[0,3,16,17,18,20],provid:[0,6,12,18,19],pure:[15,18],purpos:8,pytest:13,python3:13,python:[6,18],pytorch:[10,12,14,17,18],pytorch_multihead:12,qkv:17,qkv_bia:17,queri:[0,3,12],rand:19,randn:12,random:[0,12],randomattent:0,randomli:0,ratio:0,ravula:0,readm:13,realli:[12,20],recipi:6,recommend:13,record_rng:5,recov:16,recurs:17,redraw:0,refer:[0,10,17,20],reform:16,regist:[0,1,4,13],register_attent:[0,13],register_feedforward:1,register_positional_embed:4,registr:13,relat:[6,16],relev:[13,20],relu:[15,18],remark:13,remind:13,remov:12,ren:16,repeat:15,repetit:9,replac:[10,14,18],replace_attn_with_xformers_on:17,repo:20,report:12,repositori:[16,17],repres:0,requir:[0,12,13,16],requires_grad:12,requires_head_dimens:19,reset_peak_memory_stat:12,residu:[7,16],residual_dropout:[3,12,15,19],respons:12,result:12,reus:[17,20],revers:[2,7,9,10,14,15],reversibleblock:[5,16],reversiblesequ:[5,16],robin:16,round:12,rout:16,routingtransform:0,run:12,run_task:13,runtim:6,sai:17,same:[0,13,15,16],save:[15,16],scale:[0,6],scaleddotproduct:[0,17],search:[19,20],see:[0,1,4,6,16],self:[0,1,3,12,13,16],sens:15,septemb:18,seq:[12,19],seq_len:[0,4,12,15,19],sequenc:[0,3,12,15,16,18,19],sequenti:16,serial:9,serv:18,set:[0,6,15,16],set_rng:5,setup:6,sever:17,shape:[0,17],share:13,should:[16,19],show:17,side:0,sigmoid:18,similar:[18,19],similarli:16,simpl:12,simpli:18,sinc:20,sinepositionalembed:4,singh:0,size:[0,3,12],skip:18,slide:0,slow:6,slurm:13,sm_reg:0,small:12,smreg:0,snipper:17,snippet:13,some:[6,10,12,16,17,18,19],somebodi:20,someth:[12,13,15,16,19],sourc:[0,1,3,4,5,6,7,9,17,20],space:0,spars:[0,10,12,14],sparsifi:17,sparsiti:17,specif:[6,20],sphx:18,sputnik:6,squared_relu:18,src:9,stack:[9,16],stack_config:9,stand:0,start:[12,18],state:[10,20],still:12,stop:12,store:16,str:[0,1,4,7,9],straightforward:19,studi:20,sub:0,subclass:[0,1,4],submit:13,subsample_fract:0,suggest:6,support:[0,6,17,18],suppos:17,sure:19,sweep:19,synchron:12,take:[16,17,19],taken:15,tan:0,target:7,task:13,tensor:[0,1,3,4,5,7,9,13,16,17,18],test:[10,13,14,20],tflop:18,tgt:9,than:[6,12,15,16,18],thei:[6,8,10],them:[3,13,16,18],thi:[0,1,3,4,7,9,12,13,15,16,17,18,19,20],thing:13,think:15,three:13,through:[15,20],throughput:18,ties:13,tile:12,time:[0,12,15,18],timm:17,timmattentionwrapp:17,titl:12,to_seq_len:0,too:20,tool:[10,13],toolbox:13,toolchain:18,torch:[0,1,3,4,5,7,9,12,13,16,17,18,19],torch_cuda_arch_list:6,toward:12,tradeoff:16,train:[0,1,3,4,5,7,9],trajectori:0,tranform:18,transform:[0,7,10,14,17,18,20],transformerencod:15,transformerencoderlay:15,translat:17,transpar:6,triangular:12,trigger:[6,13],tril:12,triton:[10,12,14],tupl:7,turn:[15,16,19],tutori:[6,10,18],two:[12,16,18,19],txt:6,type:[0,13,15],typic:[0,17,18],union:[0,1,4,7,9,16],unit:[13,18],unload:6,unrel:16,urtasun:16,use:[6,12,16,17,18,19],use_razavi_pinvers:[0,13],use_separate_proj_weight:3,use_triton:7,usecas:18,used:[0,1,3,4,6,10,13,14,17,18],useful:13,uses:12,using:[6,16,18],util:0,v100:[12,18],v_skip_connect:[0,13],valu:[0,3,6,7,12,20],vanilla:7,vari:3,variabl:6,variant:[13,20],variat:17,vaswani:[0,3,16],vedaldi:0,veri:[16,18,19],verifi:16,version:6,via:[0,16],video:0,visibl:6,vision_transform:17,visiontransform:17,vit:[10,14],vocab:15,vocab_s:[4,15],vocabembed:4,wang:0,want:[6,13,17],warn:6,weight:17,welcom:20,well:[3,17],were:8,what:[0,1,4],whatev:15,when:[6,16,17,18],where:12,whether:[16,19],which:[0,1,4,6,10,12,13,15,16,17,18,19,20],window:0,window_s:0,without:16,work:[13,17,19],world_siz:13,would:[13,15,17,19],wrap:[3,16],xformer:[0,1,3,4,5,6,7,9,12,14,15,16,17,18,19],xformer_env:6,xformerblockconfig:7,xformerconfig:[9,15],xformerdecoderblock:7,xformerdecoderconfig:[7,9,16],xformerencoderblock:7,xformerencoderconfig:[7,9,16],xformerstackconfig:[9,16],xiong:0,yang:0,yet:[0,7,16],you:[0,3,6,10,12,13,14,16,17,18,19,20],your:[0,13,16,19],zaheer:0,zeng:0,zero:0,zoo:[10,14,20]},titles:["Attention mechanisms","Feedforward mechanisms","API Reference","Multi Head Attention","Position Embeddings","Reversible layer","Custom parts reference","Block factory","Factory","Model factory","Welcome to xFormers\u2019s documentation!","Tools","Using BlockSparseAttention","Extend the xFormers parts zoo","Tutorials","I\u2019m used to PyTorch Transformer Encoder, do you have an equivalent ?","Using the Reversible block","Replace all attentions from an existing ViT model with a sparse equivalent ?","Using Triton-based layers","I\u2019m only interested in testing out the attention mechanisms that are hosted here","What is xFormers?"],titleterms:{Using:[12,16,18],all:17,api:2,attent:[0,3,17,19],base:18,block:[7,16],blocksparseattent:12,build:6,cuda:6,custom:6,document:10,embed:4,encod:15,equival:[15,17],exist:17,extend:13,factori:[7,8,9],feedforward:1,from:17,fuse:18,have:15,head:3,here:19,host:19,interest:19,intro:16,kernel:6,layer:[5,18],linear:18,mechan:[0,1,19],model:[9,17],multi:3,onli:19,out:19,part:[6,13],posit:4,possibl:6,practic:16,pytorch:15,refer:[2,6],replac:17,requir:6,revers:[5,16],softmax:18,spars:[6,17],test:19,tool:11,transform:[15,16],triton:[6,18],tutori:14,usag:6,used:15,vit:17,welcom:10,what:20,xformer:[10,13,20],you:15,zoo:13}})