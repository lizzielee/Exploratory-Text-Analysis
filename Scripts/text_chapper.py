import re
import os
import sys
 
# entry of the script
def chap(novel):

    # txt book's path.
    novel_name = novel+'.txt' 
    source_path = os.getcwd()+'/'+novel_name
 
    path_pieces = os.path.split(source_path)
    novel_title = re.sub(r'(\..*$)|($)', '', path_pieces[1])
    target_path = '%s\\%s' % (path_pieces[0], novel_title)#小说分章目录
    section_re = re.compile(r'^\s*CHAPTER.*$')
    
    # create the output folder
    if not os.path.exists(target_path):
        os.mkdir(target_path)
 
    # open the source file
    input = open(source_path, 'r',encoding='utf-8')
 
    sec_count = 0
    sec_cache = []
    title_cache=[]

    output = open('%s/new.txt' % (target_path), 'w',encoding='utf-8')

    #preface_title = '%s 前言' % novel_title
    #output.writelines(preface_title)
 
        
    for line in input:
        # is a chapter's title?
        #if line.strip() == '':  #去掉空行
        #    pass
        if re.match(section_re, line):
            line = re.sub(r'\s+', ' ', line)
            print ('converting %s...' % line)
    
            output.writelines(sec_cache)

            chap_tar_path = '%s_chap_%d.txt' % (novel_name, sec_count)
            if not os.path.exists(target_path):
                os.mkdir(target_path)
            with open(chap_tar_path, 'w') as f:
                for i in sec_cache:
                    f.write(i)

            output.flush()
            output.close()
            sec_cache = []
            sec_count += 1
            #chapter_name=re.sub('(~|！+|\(+|\)+|~+|\（+|\）+|（+|!+)','_',line)
            chapter_name=re.sub('(~+|\*+|\,+|\?+|\，+|\?+)','_',line)#章节名字当文件名字时，不能有特殊符号
 
 
            # create a new section
            output = open('%s\\%s.txt' % (target_path, chapter_name), 'w',encoding='utf-8')
            output.writelines(line)
            title_cache.append(line+'\n')
        else:
            sec_cache.append(line)
            
    output.writelines(sec_cache)
    output.flush()
    output.close()
    sec_cache = []
 
    # write the menu
    output = open('%s\\menu.txt' % (target_path), 'w',encoding='utf-8')
    menu_head = '%s menu' % novel_title
    output.writelines(menu_head)
    output.writelines(title_cache)
    output.flush()
    output.close()
    inx_cache = []
    
    print ('completed. %d chapter(s) in total.' % sec_count)
 
#if __name__ == '__main__':
 #   main()

