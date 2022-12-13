#! /usr/bin/env nextflow
/*
 * Enable dsl2
 */
nextflow.enable.dsl = 2

params.cpus = "5"
params.mem = "100" //GB

process TrainTestSplit{
        publishDir "${params.publish_dir}_${params.iter}_${params.kFold}/${task.process.replaceAll(':', '_')}", mode: "copy"
        
        input:
               path clinical 
        output:
               publishDir 
               tuple file("${params.kFold}_${params.label}_samples_train.txt"),file("${params.kFold}_${params.label}_samples_test.txt"),file("${params.kFold}_Merged_clinical_${params.label}_train.tsv") 
        script:
        """
        python $baseDir/modules/train_test_split.py --label ${params.label} --clin ${clinical} --iter ${params.iter} --seed ${params.kFold}
        """

}

process DEOmics{
        publishDir "${params.publish_dir}_${params.iter}_${params.kFold}/${task.process.replaceAll(':', '_')}", mode: "copy"

        memory "${params.mem} GB"
        
        input:
                val label
                tuple file(train_samples),file(test_samples),file(clinical_train) 
                path proteome
                path metabolome 
                path transcriptome

        output:
                publishDir
                tuple file("DEMETA_in_${label}"),file("DEPRO_in_${label}"), file("DEG_in_${label}"), file("DEP_in_${label}")
        
        script:
        """
        python $baseDir/modules/DEomics.py --subgroup ${label} --clinical ${clinical_train} --omics ${metabolome} --out DEMETA_in_${label}&
        python $baseDir/modules/DEomics.py --subgroup ${label} --clinical ${clinical_train} --omics ${proteome} --out DEPRO_in_${label}&
        """
}


process metPropagate{
        publishDir "${params.publish_dir}_${params.iter}_${params.kFold}/${task.process.replaceAll(':', '_')}", mode: "copy"

        memory "${params.mem} GB"

        input:
                tuple file(DEMETA),file(DEPRO),file(DEG),file(DEP)
                path STRING_nwk_dir
                path metabolome 
                path metabolome_identifier 
                tuple file(train_samples),file(test_samples),file(clinical_train) 
        output:
                publishDir
                tuple file("Metabolome_gene_${params.label}_1"),file("Metabolome_gene_${params.label}_2")

        script:
        """ 
        mkdir -p "enrichment_files"
        
        awk -F"\t" -v cut=${params.pval} '{if(\$2<=cut&&\$3>0){print}}' ${DEMETA}> ${DEMETA}''_t1
        awk -F"\t" -v cut=${params.pval} '{if(\$2<=cut&&\$3<0){print}}' ${DEMETA}> ${DEMETA}''_t2

        grep -w -f <(cut -f 1 ${DEMETA}''_t1) ${metabolome_identifier} | cut -f 3- | sed 's/\t//g' | sort | uniq |tail -n+3 > ${DEMETA}''_1_seed
        grep -w -f <(cut -f 1 ${DEMETA}''_t2) ${metabolome_identifier} | cut -f 3- | sed 's/\t//g' | sort | uniq |tail -n+3 > ${DEMETA}''_2_seed        

        python ${params.metPropagate_dir}/DAM_cluster.py --metabolome ${metabolome} --metabolome_identifier ${metabolome_identifier} --thr 2 --clinical_data ${clinical_train} --subgroup_label ${params.label} --dir_exec ${params.metPropagate_dir} --dir_DAM "." --dir_out "."
        mkdir -p "./integration/label_files" 
        for patient in "${params.label}_1" "${params.label}_2"
        do
                Rscript ${params.metPropagate_dir}/integration/generate_labels_met_only.R \$patient ${STRING_nwk_dir}/STRING_graph_file_v11_gene_list_functional_entire_db.txt ${STRING_nwk_dir}/STRING_graph_file_v11_gene_to_nodeid_mapping_functional_entire_db.txt ./integration/label_files/STRING_\$patient

                ## parsing!
                mkdir -p "./integration/label_files"
                mkdir -p "./integration/LPA_output"
                mkdir -p "./integration/results"
                label_file="./integration/label_files/STRING_\$patient"
                intermediate_label_file_name="./integration/label_files/intermediate_label_file_\$patient"
                cut -d ' ' -f "1,2" \$label_file > \$intermediate_label_file_name
                output_file="./integration/LPA_output/\$patient"

                ## ./integration
                python2 ${params.metPropagate_dir}/label_propagation/main_graph_with_weights.py lgc -g ${STRING_nwk_dir}/STRING_graph_file_v11_functional_entire_db.txt -l \$intermediate_label_file_name -o \$output_file

                Rscript ${params.metPropagate_dir}/integration/assign_z_scores_and_interpret.R \$patient ${STRING_nwk_dir}/string_db_percent_of_fn_in_hmdb_weight.csv ${STRING_nwk_dir}/STRING_graph_file_v11_gene_to_nodeid_mapping_functional_entire_db.txt \$output_file ./integration/label_files/STRING_\$patient ./integration/results/\$patient"_STRING_ranked_label_propagation_scores.csv"
        done
        
        python ${params.metPropagate_dir}/metProp_result_normalize.py ${params.label}_1 ./integration/results/${params.label}_1_STRING_ranked_label_propagation_scores.csv 0 "Metabolome_gene_${params.label}_1"
        python ${params.metPropagate_dir}/metProp_result_normalize.py ${params.label}_2 ./integration/results/${params.label}_2_STRING_ranked_label_propagation_scores.csv 0 "Metabolome_gene_${params.label}_2"
        """
}

process seed_generation{
        publishDir "${params.publish_dir}_${params.iter}_${params.kFold}/${task.process.replaceAll(':', '_')}", mode: "copy"

        memory "${params.mem} GB"

        input:
                val k 
                val pval
                tuple file(DAMgene_group1),file(DAMgene_group2)
                tuple file(DEMETA),file(DEPRO),file(DEG),file(DEP)
        output:
                publishDir
                tuple file("PRO_DAMgene_${params.label}_1_seed"),file("PRO_DAMgene_${params.label}_2_seed")

        script:
        """
        sort -grk 2 ${DAMgene_group1}|head -n ${k}|cut -f1 > "${DAMgene_group1}_seed"
        sort -grk 2 ${DAMgene_group2}|head -n ${k}|cut -f1 > "${DAMgene_group2}_seed"

        awk -F"\t" -v cut=${pval} '{if(\$2<=cut&&\$3>0){print}}' ${DEPRO}|cut -f1 > ${DEPRO}"_1_seed"
        awk -F"\t" -v cut=${pval} '{if(\$2<=cut&&\$3<0){print}}' ${DEPRO}|cut -f1 > ${DEPRO}"_2_seed"

        cat "${DAMgene_group1}_seed" ${DEPRO}"_1_seed" > PRO_DAMgene_${params.label}_1_seed
        cat "${DAMgene_group2}_seed" ${DEPRO}"_2_seed" > PRO_DAMgene_${params.label}_2_seed  
        """ 
}


process propagation{
        publishDir "${params.publish_dir}_${params.iter}_${params.kFold}/${task.process.replaceAll(':', '_')}", mode: "copy"

        cpus params.cpus
        memory "${params.mem} GB"
        

        input:
                tuple file(seed1),file(seed2)
                path transcriptome
                path network
                path GO_graph
                path GO_gene2GO
                tuple file(train_samples),file(test_samples),file(clinical_train)
                val corrThr
        
        output:
                publishDir
                tuple file("prop_out_PRO_DAMgene_1"),file("prop_out_PRO_DAMgene_2"),file("${params.label}_transcript-transcript_nwk"),file("functional_sim_nwk")
        script:
        """
        python $baseDir/modules/filter_exp.py --transcriptome ${transcriptome} --clinical ${clinical_train} --label ${params.label} --out "transcript_exp"  
        python $baseDir/modules/instantiate_nwk.py ${network} "transcript_exp" -o "${params.label}_transcript-transcript_nwk" -nThreads 50 -corrCut ${corrThr}
        python $baseDir/modules/GO_similarity_nwk_w.py --GOgraph ${GO_graph} --gene2GO_annot ${GO_gene2GO} --templateNwk "${params.label}_transcript-transcript_nwk" --out "functional_sim_nwk" 
        python $baseDir/modules/network_propagation.py "${params.label}_transcript-transcript_nwk" "functional_sim_nwk" ${seed1} -addBidirectionEdge True --teleport_prob 0.7 -o "prop_out_PRO_DAMgene_1" 
        python $baseDir/modules/network_propagation.py "${params.label}_transcript-transcript_nwk" "functional_sim_nwk" ${seed2} -addBidirectionEdge True --teleport_prob 0.7 -o "prop_out_PRO_DAMgene_2" 
        """
}


process classification_GCN{
        publishDir "${params.publish_dir}_${params.iter}_${params.kFold}/${task.process.replaceAll(':', '_')}", mode: "copy"

        cpus params.cpus 
        memory "${params.mem} GB"


        input:
                tuple file(train_samples), file(test_samples), file(clinical_train)
                tuple file(prop_out_1), file(prop_out_2), file(inst_nwk), file(functional_sim_nwk)
                path clinical
                path transcriptome
                path methylome
                path proteome
                  
        output:
                tuple file("GNN_ourBiomarker.TransformerConv.best_model"),file("GNN_ourBiomarker.performance.txt")
        script:
        """
        python $baseDir/modules/prediction_model.py --label "${params.label}" -t ${transcriptome} -m ${methylome} -p ${proteome} -clin ${clinical} -train_samples ${train_samples} -test_samples ${test_samples} -featureSelection "ourBiomarker" -propOut1 ${prop_out_1} -propOut2 ${prop_out_2} -K ${params.K} -exp_name "GNN_ourBiomarker_teleport" -nwk ${inst_nwk}&

        """ 
}


/*
 * run pipeline
 */

workflow {
          TrainTestSplit(file(params.clinical))
          DEOmics(params.label,
                  TrainTestSplit.out,
                  file(params.proteome),
                  file(params.metabolome),
                  file(params.transcriptome))
          metPropagate(DEOmics.out,
                       params.metPropagate_STRING_dir,
                       file(params.metabolome),
                       file(params.metabolome_identifier),
                       TrainTestSplit.out)   
          seed_generation(params.k_DAM,
                          params.pval,
                          metPropagate.out,
                          DEOmics.out)
          propagation(seed_generation.out,
                      file(params.transcriptome),
                      file(params.network),
                      file(params.GO_graph),
                      file(params.GO_gene2GO), 
                      TrainTestSplit.out,
                      params.corrCut)              
          classification_GCN(TrainTestSplit.out,              
                             propagation.out,
                             file(params.clinical),
                             file(params.transcriptome),
                             file(params.methylome),
                             file(params.proteome))
}
