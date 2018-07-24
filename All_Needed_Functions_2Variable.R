factorDF <- function(a, ColumnName, m ){
  
  if (!is.null(m)){
    #a$Comment <- factor(a$Comment, levels = m )
    a[,ColumnName] <- factor(a[,ColumnName], levels = m)
    
  }
  
  #a$wafer <- factor(a$wafer, levels = unique(a$wafer[order(a$SelEpi)]))
  #a$NPtoEX <- factor(a$NPtoEX )
  #a$POR <- factor(a$POR, levels=c("POR", "Non-POR"))
 return (a)
}



###################READ DATA ################################
ReadNPN <- function(address="NPN7PAVJ2_All.csv", ColumnName='Comment', m=m){
  
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  a <- read.csv(address, header=TRUE, row.names=NULL)
  a<- factorDF(a,ColumnName, m)
  return (a)
  
}



ScatterggplotFunc <- function(DF=b, x= xstring, y= ystring, colorX=NULL, shapeX=NULL, FacetWrap=NULL, FacetScale="fixed", size=12, rotateX=F, ylog=TRUE, x_lim=c(200, 3000), y_lim=c(0, 2000), ...){
  
  require(ggplot2)
  require(scales)
  
  #p <- ggplot(DF, aes_string(x = x, y = y,  color = colorX))
  
  #color_Level <- length(levels(DF[colorX]))
  
  if( (!is.null(colorX))|(!is.null(shapeX))){
    
    if(is.null(colorX)){
      
      DF[, shapeX] <- factor( DF[, shapeX] )
      p <- ggplot(DF, aes_string(x = x, y = y,  shape = shapeX))
      shape_Level <- length(levels(DF[ ,shapeX]))
      
    }
    
    if(is.null(shapeX)){
      
      DF[, colorX] <- factor( DF[, colorX] )
      p <- ggplot(DF, aes_string(x = x, y = y,  color = colorX))
      color_Level <- length(levels(DF[ ,colorX]))
      
      
    }
    
    if( (!is.null(shapeX)) & (!is.null(colorX)))
    {
      
      DF[, colorX] <- factor( DF[, colorX] )
      DF[, shapeX] <- factor( DF[, shapeX] )
      
      p <- ggplot(DF, aes_string(x = x, y = y,  color = colorX, shape=shapeX))
      
      shape_Level <- length(levels(DF[ ,shapeX]))
      
      color_Level <- length(levels(DF[ ,colorX]))
      
    }
    
  }
  else{
    
    p <- ggplot(DF, aes_string(x = x, y = y))
    
  }
  
  if( (!is.null(FacetWrap))){
    
    p <- p + facet_wrap(as.formula(paste("~", FacetWrap)), scales=FacetScale, nrow = 1)
    
  }
  
  #p <- p + geom_boxplot(outlier.shape = NA, position = "dodge", outlier.size = 0, width=0.5)
  
  p <- p + geom_point(size = 2)
  
  p <- p + geom_line(size=1)
  
  p <- p + geom_vline(xintercept=c(0, 30, 230, 1230, 4230)  )
  
 
  if (rotateX==F){
    p <- p + theme(legend.position="bottom", text = element_text(size=size) )
  }
  else if (rotateX==T){
    
    p <- p + theme(legend.position="bottom",  text = element_text(size=size), axis.text.x = element_text(angle=90, hjust=1))
  }
  
  if (ylog == TRUE){
    
    #p <- p + scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x) ) 
    p <- p + scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                  labels = trans_format("log10", math_format(10^.x)))
  
    }
  p <- p+  coord_cartesian(xlim = x_lim, ylim=y_lim )
    
  
  #p <- p + guides(color = guide_legend(nrow = 5))
    
  if( (!is.null(colorX))|(!is.null(shapeX))){
    
    if(is.null(colorX)){
      
      p <- p + guides(shape = guide_legend(nrow = shape_Level))
      
    }
    
    if(is.null(shapeX)){
      
      p <- p + guides(color = guide_legend(nrow = color_Level))
    }
    
    if( (!is.null(shapeX)) & (!is.null(colorX)))
    {
      p <- p + guides(shape = guide_legend(nrow = shape_Level), color = guide_legend(nrow = color_Level))
      
    }
    
  }
  

  return (p)

}


LinearLinearggplotFunc <- function(DF=b, x= xstring, y= ystring, size=12, pointON=TRUE, lineON=TRUE, ...){
  
  require(ggplot2)
  
  p <- ggplot(DF, aes_string(x = x, y = y, color = "Description"))
  
  #p <- p + geom_boxplot(outlier.shape = NA, position = "dodge", outlier.size = 0, width=0.5)
  
  if (pointON==TRUE){
    
    p <- p + geom_point(size = 3)
  }
  
  if (lineON==TRUE){
    
    p <- p + geom_line(size=1)
  }
  
  p <- p + theme(legend.position="bottom", legend.title=element_blank(), text = element_text(size=size) )
  
  #p <- p+  coord_cartesian(xlim = xx[[2]], xlim = xx[[3]] )
  
  p <- p + guides(col = guide_legend(nrow = 3)) 
  
  return (p)
  
}

SaveGGPlot <- function(PlotObj = Plot_Obj, filename= fn, width=figure_width, height=figure_height,res=120, ...){
  
  png(file=filename, width=width, height=height,res=res)
  print(PlotObj)
  dev.off()
  
}



LogLinearggplotFunc <- function(DF=b, x= xstring, y= ystring, size=12, VcbON=T, lineON=T,  ...){
  
  require(ggplot2)
  
  p <- ggplot(DF, aes_string(x = x, y = y, color = "Description"))
  
  if(VcbON==T){
    p <- p + facet_wrap(~factor(Vcb))
  }
  
  
  #p <- p + geom_boxplot(outlier.shape = NA, position = "dodge", outlier.size = 0, width=0.5)

  if (VcbON==T){
    p <- p + geom_line(size=1.0, aes(linetype=factor(Nepi)))
  }
  else {
    
    if (lineON==T){
      
      p <- p + geom_line(size=1.0)
      
    }
  }
  
  p <- p + scale_x_log10(breaks=10^(-4:0))
    
  p <- p + coord_cartesian(xlim = c(10^-3, 10^-1) )
  
  p <- p + theme(legend.position="bottom", legend.title=element_blank(), text = element_text(size=size) )
  
  #p <- p+  coord_cartesian(xlim = xx[[2]], xlim = xx[[3]] )
  
  p <- p + guides(col = guide_legend(nrow = 2)) 
  
  return (p)
  
}




ACPlot <- function (po = ggplot_object, x=x_string, y=y_string, Vcb=Vcb_value, width=figure_width, height=figure_height){
  
  filename= paste(y, '_vs_', x,  '_Vcb=', toString(Vcb), 'V', '.png')
  
  png(file=filename, width=width, height=height,res=120)
  
  print(po)
  
  dev.off()
  
}

