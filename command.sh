#/bin/bash

#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --fc_num 2 --data_type cu --arch resnet101 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --fc_num 2 --data_type cu --arch resnet18 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --fc_num 2 --data_type cu --arch resnet50 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --fc_num 2 --data_type cu --arch vgg16 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --fc_num 2 --data_type cu --arch densenet169 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --fc_num 1 --data_type cu --arch resnet101 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --fc_num 3 --data_type cu --arch resnet101 --epochs 999"

#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --target_data ILSVRC --open_num 4 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --target_data ILSVRC --open_num 3 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.001 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --target_data ILSVRC --open_num 2 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.001 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --target_data ILSVRC --open_num 1 --epochs 999"

#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --target_data ILSVRC --open_num 0 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.001 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --target_data ILSVRC --open_num 0 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --target_data ILSVRC --open_num 0 --epochs 999"

#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch vgg16 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch vgg16 --use_pretrained --open_num 0 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch densenet169 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch densenet169 --use_pretrained --open_num 0 --epochs 999"

#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch resnet18 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch resnet50 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch vgg16 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch densenet169 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 3 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cub001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 1 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999"

#nsml run -d vincentcu012test -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --class_num 5 --arch vgg16 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincentcu012test -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --class_num 5  --arch densenet169 --use_pretrained --open_num 4 --epochs 999"

#nsml run -d vincent-cub002 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cub002 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch resnet18 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cub003 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cub003 -a"--batch_size 64 --optimizer sgd --lr 0.1 --fc_num 2 --data_type cu --arch resnet18 --use_pretrained --open_num 4 --epochs 999"

#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 0 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 "
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 1 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 "
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 2 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 "
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 "
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 4 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 "
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 5 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 "
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 6 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 "

#nsml run -i -d vincent-cub004 -a"--batch_size 4 --optimizer sgd --lr 0.1 --lr_sch 5 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --short_data"

#nsml run -i -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 5  --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999"



#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --milestones 300 600 900"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --milestones 200 400 600"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --milestones 150 300 450"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --milestones 250 500 750"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --milestones 100 200 300"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --milestones 100 200 400"


#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 2 --epochs 999 --milestones 300 500 700 900"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 3 --epochs 999 --milestones 300 500 700 900"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --milestones 300 500 700 900"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet18 --use_pretrained --open_num 4 --epochs 999 --milestones 300 500 700 900"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet50 --use_pretrained --open_num 4 --epochs 999 --milestones 300 500 700 900"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet152 --use_pretrained --open_num 4 --epochs 999 --milestones 300 500 700 900"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 1 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --milestones 300 500 700 900"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 3 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --milestones 300 500 700 900"


#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 2 --epochs 999 --milestones 300 500 700 900 --use_dropout"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 3 --epochs 999 --milestones 300 500 700 900 --use_dropout"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --milestones 300 500 700 900 --use_dropout"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet18 --use_pretrained --open_num 4 --epochs 999 --milestones 300 500 700 900 --use_dropout"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet50 --use_pretrained --open_num 4 --epochs 999 --milestones 300 500 700 900 --use_dropout"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 2 --data_type cu --arch resnet152 --use_pretrained --open_num 4 --epochs 999 --milestones 300 500 700 900 --use_dropout"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 1 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --milestones 300 500 700 900 --use_dropout"
#nsml run -d vincent-cub004 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 3 --fc_num 3 --data_type cu --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --milestones 300 500 700 900 --use_dropout"


#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer adam_wd --weight_decay 1e-4 --lr 0.0001 --lr_sch 0 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --use_dropout 3"
#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer adam_wd --weight_decay 1e-5 --lr 0.0001 --lr_sch 0 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --use_dropout 3"
#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer adam_wd --weight_decay 1e-6 --lr 0.0001 --lr_sch 0 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --use_dropout 3"
#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer sgd_wd --weight_decay 1e-4 --lr 0.1 --lr_sch 0 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer sgd_wd --weight_decay 1e-5 --lr 0.1 --lr_sch 0 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999"
#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer sgd_wd --weight_decay 1e-6 --lr 0.1 --lr_sch 0 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999"

#nsml run -d vincent-c3001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 2 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --use_dropout 0"
#nsml run -d vincent-c3001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --lr_sch 2 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --use_dropout 0"
#nsml run -d vincent-c3001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 2 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --use_dropout 3"

#Augmentation
#simple
#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --lr_sch 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --train_tf '[transforms.Resize((224, 224))]' --test_tf '[transforms.Resize((224, 224))]'"
#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --train_tf '[transforms.Resize((224, 224))]' --test_tf '[transforms.Resize((224, 224))]'"


#simple rotation
#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --lr_sch 0  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --train_tf '[transforms.Resize((224, 224)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((224, 224))]'"
#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --lr_sch 2 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --train_tf '[transforms.Resize((224, 224)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((224, 224))]'"

#random crop and rotation
#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --train_tf '[transforms.Resize((255, 255)),transforms.RandomCrop((224, 224)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((255, 255)),transforms.CenterCrop((224, 224))]'"

#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --lr_sch 2 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --train_tf '[transforms.Resize((255, 255)),transforms.RandomCrop((224, 224)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((255, 255)),transforms.CenterCrop((224, 224))]'"

#random crop and rotation 480
#nsml run -d vincent-cb001 -a"--batch_size 16 --optimizer adam --lr 0.0001 --lr_sch 0  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool "
#nsml run -d vincent-cb001 -a"--batch_size 16 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool "
#nsml run -d vincent-cb001 -a"--batch_size 16 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --k_fold 2"  -g 2
#nsml run -d vincent-cb001 -a"--batch_size 16 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1000 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 1"  -g 2 -c 4
#nsml run -d vincent-cb001 -a"--batch_size 16 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1000 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 4"  -g 2 -c 4
#nsml run -d vincent-cb001 -a"--batch_size 16 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1000 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 8"  -g 8 -c 8
#181, 182, 183

#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1000 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 4"  -g 4 -c 16

#nsml run -d vincent-cb001 -a"--batch_size 48 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1000 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 1"  -g 5 -c 4
# 237 without pin_memory
#nsml run -d vincent-cb001 -a"--batch_size 48 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1000 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 1"  -g 5 -c 4
# 241 with pin_memory

#nsml run -d vincent-cb001 -a"--batch_size 48 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1000 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 4"  -g 5 -c 4 --shm-size 16G
# 242 with pin_memory
#nsml run -d vincent-cb001 -a"--batch_size 48 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1000 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 8"  -g 4 -c 8 --shm-size 16G
# 245 with pin_memory

#nsml run -d vincent-cb001 -a"--batch_size 48 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1000 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 16"  -g 4 -c 12 --shm-size 32G
# 248

#nsml run -d vincent-cb001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 10 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 18"  -g 7 -c 12 --shm-size 64G
# 253

#nsml run -d vincent-cb001 -a"--adaptive_pool --use_multicrop --batch_size 4 --optimizer adam --lr 0.0001 --lr_sch 2 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.FiveCrop(480)]'"

#random crop and rotation 480
#nsml run -d vincent-cb001 -a"--batch_size 16 --optimizer adam --lr 0.0001 --lr_sch 0  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool "



#nsml run -d vincent-c3001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 2 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --use_dropout 0"
#nsml run -d vincent-c3001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --lr_sch 2 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --use_dropout 0"
#nsml run -d vincent-c3001 -a"--batch_size 64 --optimizer sgd --lr 0.1 --lr_sch 2 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --use_dropout 3"
#nsml run -d vincent-c3001 -a"--batch_size 64 --optimizer adam --lr 0.0001 --lr_sch 2 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --use_dropout 3"

nsml run -d vincent-cb001 -a"--adaptive_pool --use_multicrop --batch_size 4 --optimizer adam --lr 0.0001 --lr_sch 2 --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 999 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.FiveCrop(480)]'"
nsml run -d vincent-cb001 -a"--batch_size 16 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 16" -g 5 -c 4 --shm-size 16G
nsml run -d vincent-cb001  -g 1 -c 4 --shm-size 16G  -a"--batch_size 16 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1000 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 4"
nsml run -d vincent-cb001  -g 5 -c 8 --shm-size 32G --queue -a"--batch_size 48 --optimizer adam --lr 0.0001 --lr_sch 2  --fc_num 2 --data_type c --arch resnet101 --use_pretrained --open_num 4 --epochs 1000 --train_tf '[transforms.Resize((512, 512)),transforms.RandomCrop((480, 480)),transforms.RandomRotation(20)]' --test_tf '[transforms.Resize((512, 512)),transforms.CenterCrop((480, 480))]' --adaptive_pool --num_workers 8"