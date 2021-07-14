DATA = read.csv('/data/contact/dec2020/processed.csv')
#FIGS = '/data/figs/contact/'
FIGS = '/data/contact/dec2020/figs/'
#DATA = read.csv('/Users/magnuslynch/Work/Non Current/Research/Contact/feb2020/processed.csv')
#FIGS = '/Users/magnuslynch/Work/Dropbox/Research/Lab/projects/contact_ai/figs/'
source('common.r')

library(ggplot2)
library(e1071) 
library(plotROC)
library(pROC)
test <- data.frame(dose=c("D0.5", "D1", "D2"),len=c(4.2, 10, 29.5))
p <- ggplot(data=test,aes(x=dose,y=len)) + geom_bar(stat="identity")


#Filter dataset to remove rows where 
DATA.diag <- DATA[!is.na(DATA$allergy_code),]
#DATA.diag.complete <- 0
#DATA.diag.complete.test <- 0
#DATA.diag.complete.train <- 0
#DATA.nodiag <- DATA[is.na(DATA$allergy_code),]
#OUTCOME <- 'allergy_now'
#OUTCOME <- 'LACTONE'
#OUTCOME <- 'PRIMIN'
set_outcome <- function(outcome_)
{
    # need to use <<- to set global variable
    #DATA.diag$outcome <- DATA.diag[,OUTCOME]
    print(outcome_)
    DATA.diag$outcome <<- DATA.diag[,outcome_]
    DATA.diag$outcome[DATA.diag$outcome==-1] <<- NA
    DATA.diag$outcome[DATA.diag$outcome==0] <<- F
    DATA.diag$outcome[DATA.diag$outcome==1] <<- T
    pred_vars = c('outcome','id','age','date','sex','site','occupation','duration','spread','atopy','psoriasis','housework','fh_atopy')
    DATA.diag.complete <<- DATA.diag[,pred_vars]
    DATA.diag.complete <<- DATA.diag.complete[complete.cases(DATA.diag.complete),]
    df <- DATA.diag.complete[order(DATA.diag.complete$id),]
    DATA.diag.complete.train <<- df[1:(nrow(df)*0.8),]
    DATA.diag.complete.test <<- df[(nrow(df)*0.8):nrow(df),]
    DATA.diag.complete.test <<- DATA.diag.complete.test[DATA.diag.complete.test$occupation != 'electroplating',]
    DATA.diag.complete.test <<- DATA.diag.complete.test[DATA.diag.complete.test$occupation != 'military',]
    return(DATA)
}



#≠=================================================================================
# PLot demographics

# show the proportion that have diagnosis

plot_diag <- function()
{
    df<-data.frame("Diagnosis",nrow(DATA.diag))
    names(df)<-c("group","count")
    de<-data.frame("No diagnosis",nrow(DATA.nodiag))
    names(de)<-c("group","count")
    df <- rbind(df, de)
    print(df)

    bp<- ggplot(df, aes(x="", y=count, fill=group)) +
        geom_bar(width = 1, stat = "identity",color="white") +
        coord_polar("y", start=0) + 
        theme_void()
    ggsave(paste(FIGS,'demographics/diagnosis.pdf',sep=''))
}

#sex, spread, atopy, housework

plot_piecharts <- function()
{
    piechart <- function(colname,filename)
    {
        tab <- table(DATA.diag[colname])
        tab <- as.data.frame(tab)
        colnames(tab) <- c('group','freq')
        bp<- ggplot(tab, aes(x="", y=freq, fill=group)) +
            geom_bar(width = 1, stat = "identity",color="white") +
            coord_polar("y", start=0) + 
            theme_void()
        ggsave(filename)
    }


    for(colname in c('sex','spread','atopy','housework')) {
        print(colname)
        piechart(colname,paste(FIGS,'demographics/',colname,'.pdf',sep=''))
    }
}


#occupation, site

plot_ordered_barcharts <- function()
{
    ordered_barchart <- function(colname,filename)
    {
        tab <- table(DATA.diag[colname])
        tab <- as.data.frame(tab)
        colnames(tab) <- c('group','count')
        print(tab)

        p <- ggplot(tab, aes(x=reorder(group,-count),y=count))+
            geom_bar(stat="identity" , fill="steelblue")+
            theme_minimal()+
            theme(axis.text.x = element_text(angle = 90,vjust=0.5,hjust=1 ))
        
        ggsave(filename)
        return(p)
    }

    
    for(colname in c('occupation','site')) {
        print(colname)
        ordered_barchart(colname,paste(FIGS,'demographics/',colname,'.pdf',sep=''))
    }
}

#duration

plot_duration <- function()
{
    DATA.diag$duration <- factor(DATA.diag$duration,levels = c('less_month','1_to_6_months','6_months_to_year','1_to_5_years','more_5_years'))
    p <- ggplot(DATA.diag,aes(x=factor(duration)))+
        geom_bar(stat="count" , fill="steelblue")+
        theme_minimal()
    
    ggsave(paste(FIGS,'demographics/duration.pdf',sep=''))
}

#age 

plot_age <- function()
{
	p <- ggplot(DATA.diag, aes(x=age)) + 
        geom_histogram(binwidth=5,fill="steelblue",color="steelblue")
    
    ggsave(paste(FIGS,'demographics/age.pdf',sep=''))
}


plot_codes <- function()
{
    df<-data.frame("allergy_now",as.numeric(table(DATA.diag$allergy_now)['Y']))
    names(df)<-c("code","count")
    de<-data.frame("allergy_past",as.numeric(table(DATA.diag$allergy_past)['Y']))
    names(de)<-c("code","count")
    df <- rbind(df, de)
    de<-data.frame("allergy_unexplained",as.numeric(table(DATA.diag$allergy_unexplained)['Y']))
    names(de)<-c("code","count")
    df <- rbind(df, de)
    de<-data.frame("allergy_missed",as.numeric(table(DATA.diag$allergy_missed)['Y']))
    names(de)<-c("code","count")
    df <- rbind(df, de)
    de<-data.frame("allergy_photosensitivity",as.numeric(table(DATA.diag$allergy_photosensitivity)['Y']))
    names(de)<-c("code","count")
    df <- rbind(df, de)
    
    p <- ggplot(df,aes(x=code,y=count))+
        geom_bar(stat="identity" , fill="steelblue")+
        theme_minimal()

    ggsave(paste(FIGS,'demographics/codes.pdf',sep=''))
}


#Generate a heatmap of the relationship beteen
#diagnosis code and allergen


heatmap_code_count <- function()
{
    codes <- c('allergy_now','allergy_past','allergy_unexplained')
	exclude <- c('MI','SODIUM.METABISULFITE','LYRAL','CHROM375','BETAINES')
	allergens <- as.character(PRETTY$Short)
	allergens <- allergens[!allergens %in% exclude] 
    v <- integer(length(allergens))
    results <- data.frame(allergen=allergens,
                          allergy_now=v,allergy_past=v,allergy_unexplained=v)
    rownames(results) <- results$allergen
    results$allergen <- NULL

    for(i in 1:nrow(DATA.diag)) {
        print(i)
        code = ''
        for(x in codes) {
            if(DATA.diag[i,x]=='Y') code = x
        }
        if(code=='') next 
        
        for(a in as.character(rownames(results))) {
			if(a %in% exclude) next
            x = DATA.diag[i,a]
            if(!is.null(x) && x==1) results[a,code] = results[a,code]+1
        } 
    }
    return(results) 
}


heatmap_code_plot <- function(results)
{
	new <- data.frame()
	for (row in rev(rownames(results))) {
		for(col in colnames(results)) {
            x <- min(results[row,col],200)
			test <- pretty_name(row)
			new <- rbind(new,data.frame(allergen=row,diagnosis=col,val=x,pretty=test))
		}		
	}

	base_size <- 12
	p <- ggplot(data=new, aes(x=diagnosis,y=pretty,fill=val)) + 
		geom_tile(colour = "white") + 
		scale_fill_gradient(low = "white",high = "steelblue") +
		theme_grey(base_size = base_size) + labs(x = "",y = "") + 
		scale_x_discrete(expand = c(0, 0)) +
		scale_y_discrete(expand = c(0, 0)) + 
		theme(axis.ticks.x=element_blank())+
		theme(axis.ticks.y=element_blank())

    ggsave(paste(FIGS,'demographics/heatmap_code.pdf',sep=''))

}


#≠=================================================================================
# Logistic regression

lr_fit <- function(df)
{
    model <- glm(outcome ~ age+sex+date+site+occupation+duration+spread+atopy+psoriasis+housework+fh_atopy, data = df, family = "binomial")
    return(model)
}


lr_stats <- function()
{
    # calculate stats and relationship to predictor variables for entire dataset
    lr_model = lr_fit(DATA.diag.complete)
    print(summary(lr_model))
    #write.csv(tidy(lr_model),paste(FIGS,'logistic/lr_model.csv',sep=''))
    
    #takes a long time to calc confidence intervals
    #lr_conf = confint(lr_model) 
    #write.csv(lr_conf,paste(FIGS,'logistic/lr_conf.csv',sep=''))
    #print(summary(lr_conf))
    return(lr_model)
}

lr_selected_stats <- function()
{
    results <- data.frame(matrix(ncol=7,nrow=0))
    #names(results) <- c('Predicted','Parameter','Magnitude','P Value','P Value corrected')
    for(outcome in c('allergy_now','one_pos','LACTONE','CARBA','NICKEL','EPOXY','GERMALL')) {
        set_outcome(outcome)
        model <- lr_stats()
        #print(outcome)
        sig <- summary(model)$coeff[-1,4] < 0.05 
        sig <- names(sig)[sig==TRUE]
        conf_ints <- data.frame(confint(model))

        for(x in sig) {
            magnitude <- as.numeric(coef(summary(model))[,1][x])
            odds_ratio <- exp(magnitude)
            ci_lower <- exp(conf_ints[x,'X2.5..'])
            ci_upper <- exp(conf_ints[x,'X97.5..'])
            p_value <- as.numeric(coef(summary(model))[,4][x])
            if(p_value < 0.05/76) p_value_corr <- '*'
            else p_value_corr <- ' ' 
            new <- data.frame(outcome,x,magnitude,odds_ratio,ci_lower,ci_upper,p_value,p_value_corr)
            names(new) <- c('Predicted','Parameter','Magnitude','Odds Ratio','CI lower','CI upper','P Value','P Value Corrected')
            results <- rbind(results,new)
            #print(c(outcome,x,magnitude,p_value))
        }
    }
    print(results)
    
    write.csv(format(results,digits=2,nsmall=2),paste(FIGS,'sig_results.csv',sep=''))
    return(results)
}





