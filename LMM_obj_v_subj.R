##### IMPORTS AND PACKAGES#####
install.packages("lme4")
library(lme4)
install.packages("lmerTest")
library(lmerTest)
install.packages("ggnewscale")
library(ggnewscale)
install.packages("tidyverse")
library(tidyverse)
install.packages("effects")
library(effects)
install.packages("gridExtra")
library(gridExtra)
install.packages("sjPlot")
library(sjPlot)
install.packages("ggcorrplot")
library(ggcorrplot)
install.packages("GGally")
library(GGally)
install.packages("lavaan")
library(lavaan)
install.packages("semPlot")
library(semPlot)



############### READ IN DATA ###############
updated_data_directory = "/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/day-to-day-modeling-files-updated-data/"
setwd(updated_data_directory)
obj_v_subj_dt <- read.csv("MASTER_obj_subj_file_MASTER_filled_FINAL.csv", header = TRUE)


########## CONFIRMATORY FACTOR ANALYSIS ##########
# set up model
CFA.model <- 'subjective sleep quality  =~ Mentally.Alert + Deep.Sleep + Overall.quality + Well.rested'

# fit model to our data
fit <- cfa(CFA.model, data = obj_v_subj_dt)

# view the summary of the model
summary(fit, fit.measures = TRUE, standardized = T)

# plot the model
semPaths(fit, "std", edge.label.cex = 0.9, curvePivot = TRUE, sizeLat = 14, sizeLat2 = 14,
         nodeLabels = c("Mentally alert", "Deep sleep", "Overall quality", "Well rested", "Subj. sleep quality"), 
         label.cex = 1.2, 
         sizeMan = 10, 
         )


########## DATA PROCESSING ##########
# add model fit as column
temp <- predict(fit)
obj_v_subj_dt$latent_variable <- temp

# filter out all values with WASO > 4 hours (considered extraneous)
filtered_data <- obj_v_subj_dt %>% 
  filter(WASO <= 4)

num_filtered <- nrow(obj_v_subj_dt) - nrow(filtered_data)

print(paste("Number of values with WASO > 4:", num_filtered))

obj_v_subj_dt <- filtered_data

# Etiology cleaning
obj_v_subj_dt$Etiology <- factor(obj_v_subj_dt$Etiology, levels= c("HC", "AD", "LB"))
obj_v_subj_dt <- subset(obj_v_subj_dt, Etiology != "Other")

# impairment grouping/cleaning
obj_v_subj_dt$Impaired <- factor(obj_v_subj_dt$Impaired, levels = c("Normal", "MCI", "Dementia"))

# mutate obj_v_subj_dt to add MoCA bins
obj_v_subj_dt <- obj_v_subj_dt %>%
  mutate(MoCA_binned = case_when(
    MoCA < 20 ~ "low(<20)",
    MoCA >= 20 & MoCA <= 25 ~ "intermediate(20-25)",
    MoCA > 25 ~ "high(>25)",
    TRUE ~ "unknown"  # not all subjects have MoCA data available
  ))

# make sure that high MoCA is the default for regression comparisons: cleaning
obj_v_subj_dt$MoCA_binned <- factor(obj_v_subj_dt$MoCA_binned, levels = c("high(>25)", "intermediate(20-25)", "low(<20)"))

########## DO THE PARTIAL POOLING ##########
# VARIABLE KEY: sleep duration = SleepDurationInSpt, Wake After Sleep Onset = WASO. Change variables in the models below accordingly

# 1. run for NO predictor 
# pp_mod <- lmer(Well.rested ~ Etiology + Age + Sex + (1 | Study.ID), data = obj_v_subj_dt) 
#summary(pp_mod)

#2. run for predictor
# model for ETIOLOGY:
pp_mod <- lmer(latent_variable ~ WASO*Etiology + Age + Sex + (1 | Study.ID), data = obj_v_subj_dt) # For impairment model: change Etiology to Impaired, add etiology as a covariate, and change data to obj_v_subj_dt. For etiology, change data to obj_v_subj_dt, get rid of etiology as a covariate. 
summary(pp_mod)

# model for IMPAIRMENT: 
#pp_mod <- lmer(latent_variable ~ WASO*Impaired + Age + Sex + (1 | Study.ID), data = obj_v_subj_dt)
#summary(pp_mod)

# model for MoCA: 
#pp_mod <- lmer(latent_variable ~ WASO*MoCA_binned + Age + Sex + (1 | Study.ID), data = obj_v_subj_dt)
#summary(pp_mod)

#3. Age model (with no etiology) 
#pp_mod_age <- lmer(latent_variable ~ WASO * Age + Sex + (1 | Study.ID), data = obj_v_subj_dt)
#summary(pp_mod_age)

# 4. Healthy individuals only
obj_v_subj_dt_HC_only <- subset(obj_v_subj_dt, Etiology == "HC")
#pp_mod_age_HC_only <- lmer(latent_variable ~ WASO * Age + Sex + (1 | Study.ID), data = obj_v_subj_dt_HC_only)
#summary(pp_mod_age_HC_only)

# 5. Quadratic model
pp_mod_quad <- lmer(Overall.quality ~ poly(SleepDurationInSpt, 2, raw = TRUE)^2*Etiology + Age + Sex + (1 | Study.ID), data = obj_v_subj_dt_3)
pp_mod_summary_quad <-  summary(pp_mod_quad)


########## get the partial pooling effects of etiology on objective variable, to plot. Whatever is in parenthesis should be what we are grouping by, and the objective variable. dont worry about warning message here ##########
#1. Etiology
pp <- data.frame(Effect(c("Etiology", "WASO"), pp_mod)) 
#2. Impairement
pp <- data.frame(Effect(c("Impaired", "SleepDurationInSpt"), pp_mod)) 
#3. MoCA
pp <- data.frame(Effect(c("MoCA_binned", "SleepDurationInSpt"), pp_mod))  
#4. Quadratic
pp_quad <- data.frame(Effect(c("Etiology", "SleepDurationInSpt"), pp_mod_quad))


##########  PLOTTING ########## 
# 1. plot the model - ETIOLOGY
ggplot(pp, aes(x = WASO, y = fit, color = factor(Etiology), group = Etiology)) + 
  geom_line(linewidth = 2) +
  geom_ribbon(aes(fill = factor(Etiology), ymin = lower, ymax = upper), alpha = 0.1) + 
  labs(x = "WASO (hours)", y = "Subjective Quality", color = "Etiology") +
  theme_classic(base_size = 30) + 
  scale_color_manual(name = "Etiology", values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) +
  scale_fill_manual(name = "Etiology", values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A"))  

# save plot
ggsave("/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/figs/WASO_subjsleep.png", width = 7.5, height = 6.25, dpi = 300)

### Same plot as above, but with scattered points on top  ###
ggplot(pp, aes(x = WASO, y = fit, color = factor(Etiology), group = Etiology)) + 
  geom_line(linewidth = 2) +
  geom_ribbon(aes(fill = factor(Etiology), ymin = lower, ymax = upper), alpha = 0.1) + 
  labs(x = "WASO (hours)", y = "Subjective Quality", color = "Etiology") +
  theme_classic(base_size = 30) + 
  scale_color_manual(name = "Etiology", values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) +
  scale_fill_manual(name = "Etiology", values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) +
  geom_point(data = obj_v_subj_dt, aes(x = WASO, y = latent_variable, color=factor(Etiology)), size = 3, shape = 16, inherit.aes = FALSE, alpha = 0.5)


# 2. plot the model - IMPAIRMENT
ggplot(pp, aes(x = SleepDurationInSpt, y = fit, color = factor(Impaired), group = Impaired)) + 
  geom_line(size = 2) +
  geom_ribbon(aes(fill = factor(Impaired), ymin = lower, ymax = upper), alpha = 0.1) + 
  labs(x = "Sleep Duration (Hours)", y = "Subjective Quality", color = "Impaired") +
  theme_classic(base_size = 20) + 
  scale_color_manual(name = "Impaired", values = c("Dementia" = "#E41A1C", "MCI" = "#377EB8", "Normal" = "#4DAF4A")) +
  scale_fill_manual(name = "Impaired", values = c("Dementia" = "#E41A1C", "MCI" = "#377EB8", "Normal" = "#4DAF4A"))

# save plot 
ggsave("/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/figs/SleepDuration_subjsleep_IMPAIRMENT.png", width = 7.5, height = 6.25, dpi = 300)

# 3. plot the model - MOCA
ggplot(pp, aes(x = SleepDurationInSpt, y = fit, color = factor(MoCA_binned), group = MoCA_binned)) + 
  geom_line(linewidth = 2) +
  geom_ribbon(aes(fill = factor(MoCA_binned), ymin = lower, ymax = upper), alpha = 0.1) + 
  labs(x = "Sleep Duration (hours)", y = "Subjective Quality", color = "MoCA score") +
  theme_classic(base_size = 30) + 
  scale_color_manual(name = "MoCA score", values = c("high(>25)" = "#4DAF4A", "intermediate(20-25)" = "#377EB8", "low(<20)" = "#E41A1C")) +
  scale_fill_manual(name = "MoCA score", values = c("high(>25)" = "#4DAF4A", "intermediate(20-25)" = "#377EB8", "low(<20)" = "#E41A1C")) 

# save plot
ggsave("/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/figs/SleepDuration_subjsleep_MoCA.png", width = 9.5, height = 6.25, dpi = 300)


########## CORRELATION MATRIX AND VISUALIZATION ##########
# NOTE: this code isn't really in use, but has some potentially useful visualizations

# set up the matrix
subset_data_for_cor_matrix <- obj_v_subj_dt %>% select(Well.rested, Overall.quality, Deep.Sleep, Mentally.Alert)
cor <- round(cor(subset_data_for_cor_matrix, use = "pairwise.complete.obs"), 1)
# Create the ggcorrplot with a custom color scale from 0 to 1
ggcorrplot(cor, lab = TRUE, lab_size = 6) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 20),  # Adjust the text size for x-axis labels
    axis.text.y = element_text(size = 20),  # Adjust the text size for y-axis labels
    legend.text = element_text(size = 15),  # Adjust the legend text size
    legend.title = element_text(size = 16)  # Adjust the legend title size
  )


########## RESIDUALS, SLOPES ANALYSIS ##########
# NOTE: same as above, code is also not really in use

# extract residuals from the model. 
filtered_data$WASO_res <- resid(lm(WASO~Age+Sex, data=filtered_data)) 
filtered_data$Well.rested_res <- resid(lm(Well.rested~Age+Sex, data=filtered_data)) 

# get a data frame only having a unique data table 
unique_filtered_data <- obj_v_subj_dt %>% distinct(Study.ID, .keep_all = TRUE)

### Extract and plot individual slopes ###
lme.model <- lmer(Well.rested ~ WASO + (WASO|Study.ID), data=obj_v_subj_dt_3)  
coef.model <- coef(lme.model)$Study.ID
# add the slopes to the filtered data df 
unique_filtered_data$slopes <- coef.model$WASO

# plot the histogram 
ggplot(unique_filtered_data, aes(x = slopes)) + 
  geom_histogram(color = "black", fill = "white", size =1) + 
  labs(x = "Slope", y = "Count", title = "Slope of WASO vs. Well rested") + 
  theme_classic(base_size = 20) +
  theme(plot.title = element_text(hjust = 0.5)) 

# create a box plot for each etiology, with etiology (HC, LB, AD) on the x-axis 
# and slope on the y-axis

ggplot(unique_filtered_data, aes(x = Etiology, y = slopes, color = factor(Etiology))) + 
  geom_boxplot(size = 1) + 
  geom_point() +
  theme_classic(base_size = 20) + 
  labs(x = 'Etiology', y = 'Slope', title = 'WASO vs. Well rested', color = "Etiology") + 
  theme(plot.title = element_text(hjust = 0.5)) 


########## PLOT BOX PLOT OF BASELINE MEASURES. NOTE - this is Figure 1a in the paper draft ##########

# filter out etiology = other
obj_v_subj_dt_3 <- subset(obj_v_subj_dt, Etiology != "Other")

# filter unique_filtered_data so it has a mean value for WASO for EACH USER
# Group by 'Study.ID' and calculate the mean of 'WASO'
df_one_mean_subj_per_ID <- obj_v_subj_dt_3 %>%
  group_by(Study.ID) %>%
  summarise(mean_sleep_duration = mean(SleepDurationInSpt, na.rm = TRUE))

result_df <- right_join(df_one_mean_subj_per_ID, unique_filtered_data, by = "Study.ID")


# Make a box plot of each objective and subjective measure, by etiology 
ggplot(result_df, aes(x = Etiology, y = WASO, color = factor(Etiology))) + 
  geom_boxplot(size = 1) + 
  scale_color_manual(name = "Etiology", values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) + 
  theme_classic(base_size = 20) + 
  labs(x = 'Etiology', y = 'WASO (hours)', color = "Etiology")  
  

# calculate the mean by 'Study.ID' first
mean_by_study <- aggregate(obj_v_subj_dt_3[, c("Well.rested", "Deep.Sleep", "Mentally.Alert", "Overall.quality")],
                           by = list(obj_v_subj_dt_3$Study.ID),
                           mean,
                           na.rm = TRUE)

# Rename the "Group.1" column to "Study.ID"
colnames(mean_by_study)[1] <- "Study.ID"

# merge the dfs, based on Study ID
merged_data <- merge(unique_filtered_data, mean_by_study, by = "Study.ID")

# Convert the mean_data to long format
mean_data_long <- tidyr::gather(mean_data, key = "SubjectiveMeasure", value = "MeanValue", -Group.1)

# Calculate standard deviation
sd_data <- aggregate(mean_by_study[, 2:5],
                     by = list(merged_data$Etiology),
                     sd,
                     na.rm = TRUE)

# Convert standard deviation data to long format
sd_data_long <- tidyr::gather(sd_data, key = "SubjectiveMeasure", value = "SDValue", -Group.1)

# Merge mean and standard deviation data
plot_data <- merge(mean_data_long, sd_data_long, by = c("Group.1", "SubjectiveMeasure"))

# Custom labels for subjective measures
custom_labels <- c("Mentally.Alert" = "Mentally Alert",
                   "Deep.Sleep" = "Deep Sleep",
                   "Overall.quality" = "Overall Quality",
                   "Well.rested" = "Well Rested")

# Create label data (one row per subjective measure)
label_data <- plot_data %>%
  distinct(SubjectiveMeasure) %>%
  mutate(Label = custom_labels[SubjectiveMeasure],
         y = -0.3)  # BELOW y=0 axis

ggplot(plot_data, aes(x = SubjectiveMeasure, y = MeanValue, fill = factor(Group.1))) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  
  geom_errorbar(aes(ymin = MeanValue - SDValue, ymax = MeanValue + SDValue),
                position = position_dodge(width = 0.8),
                width = 0.2,
                linewidth = 0.9) +
  
  # Add labels BELOW x-axis
  geom_text(data = label_data,
            aes(x = SubjectiveMeasure, y = y, label = Label),
            inherit.aes = FALSE,
            vjust = 1,
            size = 4.5) +  # smaller text size
  
  scale_x_discrete(labels = NULL) +  # hide default x labels
  scale_y_continuous(expand = c(0, 0)) +
  
  # Expand view BELOW y=0 line and allow labels to show outside
  coord_cartesian(ylim = c(0, 5), clip = "off") +
  
  theme_classic(base_size = 24) +
  labs(x = "", y = "Mean Rating", fill = "Etiology") +
  scale_fill_manual(values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.border = element_blank(),
    plot.margin = margin(t = 10, r = 10, b = 50, l = 10)  
  )

# save the plot 
ggsave("/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/figs/subjective_measures_bar_plot.png", width = 8.5, height = 6.25, dpi = 300)




########## SAME THING AS ABOVE (SUBJECTIVE VS. SCORE, for all etiologies), BUT WITH A BOXPLOT ##########
# Custom labels for the subjective measures
custom_labels <- c("Mentally Alert" = "Mentally.Alert",
                   "Deep Sleep" = "Deep.Sleep",
                   "Overall Quality" = "Overall.quality",
                   "Well Rested" = "Well.rested")

# Convert the merged_data to a long format for plotting
merged_data_long <- tidyr::gather(merged_data, 
                                  key = "SubjectiveMeasure", 
                                  value = "Value", 
                                  Well.rested.x, Deep.Sleep.x, Mentally.Alert.x, Overall.quality.x)

# Plot the box plot with jittered points
ggplot(merged_data_long, aes(x = SubjectiveMeasure, y = Value, fill = factor(Etiology))) +
  # Add the box plot with transparent fill
  geom_boxplot(outlier.shape = NA, width = 0.7, position = position_dodge(width = 0.8), alpha = 0.3) +
  # Add jittered points on top of the box plot
  geom_jitter(
    aes(color = factor(Etiology)),
    position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
    size = 2,
    alpha = 0.6
  ) +
  # Adjust x-axis labels with custom labels
  scale_x_discrete(labels = custom_labels) +
  # Set y-axis limits and configure scaling
  scale_y_continuous(limits = c(0, 7), expand = c(0, 0)) +
  # Set the theme and label aesthetics
  theme_classic(base_size = 24) +
  labs(x = "", y = "Score", fill = "Etiology", color = "Etiology") +
  # Set custom colors for fill and points
  scale_fill_manual(values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) +
  scale_color_manual(values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) +
  # Customize theme elements
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.ticks.x = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.border = element_blank()
  )



########## OBJECTIVE VARIABLE VS. SUBJ. SCORE, FOR ALL ETIOLOGIES. NOTE: this is Figure 2 in the paper draft ##########
# define custom x axis labels
custom_labels_obj <- c("WASO" = "WASO",
                   "SleepDurationInSpt" = "Sleep Duration")

# Convert the merged_data to a long format for plotting
merged_data_long_obj <- tidyr::gather(merged_data, 
                                  key = "ObjectiveMeasure", 
                                  value = "Value", 
                                  WASO, SleepDurationInSpt)

# Subset the data for each measure
waso_data <- subset(merged_data_long_obj, ObjectiveMeasure == "WASO")
sleep_data <- subset(merged_data_long_obj, ObjectiveMeasure == "SleepDurationInSpt")  # or whatever label you use for sleep duration

# Plot 1: WASO
plot_waso <- ggplot(waso_data, aes(x = "WASO", y = Value, fill = factor(Etiology))) +
  geom_boxplot(outlier.shape = NA, width = 0.7, alpha = 0.3,
               position = position_dodge(width = 0.8)) +
  geom_jitter(
    aes(color = factor(Etiology)),
    position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
    size = 2,
    alpha = 0.6
  ) +
  scale_y_continuous(limits = c(-0.1, 5), expand = c(0, 0)) +
  theme_classic(base_size = 24) +
  labs(x = "", y = "Hours", fill = "Etiology", color = "Etiology") +
  scale_fill_manual(values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) +
  scale_color_manual(values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.ticks.x = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.border = element_blank()
  )

# Plot 2: Sleep duration
plot_sleep <- ggplot(sleep_data, aes(x = "Sleep Duration", y = Value, fill = factor(Etiology))) +
  geom_boxplot(outlier.shape = NA, width = 0.7, alpha = 0.3,
               position = position_dodge(width = 0.8)) +
  geom_jitter(
    aes(color = factor(Etiology)),
    position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
    size = 2,
    alpha = 0.6
  ) +
  scale_y_continuous(limits = c(-0.1, 11), expand = c(0, 0)) +
  theme_classic(base_size = 24) +
  labs(x = "", y = "Hours", fill = "Etiology", color = "Etiology") +
  scale_fill_manual(values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) +
  scale_color_manual(values = c("AD" = "#E41A1C", "LB" = "#377EB8", "HC" = "#4DAF4A")) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.ticks.x = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.border = element_blank()
  )

# Combine the plots side by side
plot_waso + plot_sleep + plot_layout(ncol = 2, guides = "collect")

# save plot
ggsave("/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/figs/box_plot_with_jitter.png", width = 8.5, height = 6.25, dpi = 300)












