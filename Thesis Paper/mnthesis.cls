%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mnthesis.cls: mnthesis document class (Latex)
% Les Clowney ----- modified by K. Thorne for UofMinn
%
%  5/12/89   make version for University of Minnesota
%  11/23/89  modifications for 11-pt version for UofM
%  2/14/90   correct entries to table of contents
%  3/25/94   add in signature and abstract signature pages. -Jeff Nelson
%  5/4/2005  Modify to latex2e class format from old style format - BPS
%  5/24/2007 Fix the title and abstract page formatting. - BPS
% 11/17/2009 Fixed ordering of acknowledgments and abstract as required - PP
% 11/17/2009 'References' added to Table of Contents as required - PP
% 11/17/2009 Note: Table of content and list of Figures are ok in 'draft' format, not in 'final' (which is default) - PP
% 5/21/2010 Added Dedication - EB
% 5/24/2010 Added section to correctly give M.S. thesis or Ph.D. dissertation - EB
% 11/23/2010 Fixed the issue with restarting the numbering of pages if both Acknowledgement and Dedication pages are present
%            If Dedication page alone is present please uncomment line 406   - Sriram Doraiswamy
% 11/23/2010 Fixed the problem where additional pages of Acknowledgement were not correctly numbered - Sriram Doraiswamy
% 11/25/2010 Name of the advisor and Master of Arts or Science appears properly - Sriram Doraiswamy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summary of New Commands (many are not used outside of the style file)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The following are primarily for the titlepage et al....
%\draft			(this invokes the macro \draftrue)
%\copyrightpage		(make a copyright page)
%\figurespage		(make a page for list of figures)
%\tablespage		(make a page for list of tables)
%\title{thesis title}	(this is defined by the report style)
%\author{author's name}	(this is defined by the report style)
%\phd			(if invoked then \phdtrue is set.)
%\degree{degree type}	(Default "Master of Science")
%
%\words{words in abstract}
%\abstract{the body of the abstract}
%\acknowledgements{the body of the acknowledgments}
%\dedication{the body of the dedication}
%\director{the principal advisor's name}
%\submissionyear{year of submission to the Graduate School}
%\submissionmonth{month of submission to the Graduate School}
%(The default dates used will be those at which the document was prepared)
%\begin-end{vita}  (begin single spacing after this point for the vita)
%
%       ******* Booleans *******
%\ifpagestyletopright	(invoke \pagestyle{topright})
%\ifdraft		(Will do some things differently if draft. Set by \draft)
%\ifcopyright		(Add a copyright notice? (Set by \copyrightpage)
%\ifabstract		(set if command \abstract invoked)
%\ifpreface		(set if command \preface invoked)
%\ifextra		(set if command \extra invoked)
%\ifacknowledgements	(set by \acknowledgements)
%\ifabstract		(set by \abstract)
%\iffigures
%\iftables
%\ifafterpreface (afterpreface sections pagenumber must be at topright
%               corner. If user has chosen a header then it must be overridden.)
%               (Produce a List of figures? The default is to do so.)
%\tablestrue
%               (Produce a List of tables?)
%\begin/end{descriptionlist} (Basically a modified  \description.)
%\fullpagefigure Creates a figure where the page is a vbox whose
%                height is \textheight.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Redefine the way that LaTeX starts up so that its simpler to use.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\input{report.cls}\relax
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PAGE LAYOUT
% We need 1" (~72pt) margins except on binding edge, where it is 1 1/2".
% They are a bit larger to handle lines with overfull boxes.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% My Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\setlength{\paperheight}{11.0in}
\setlength{\paperwidth}{8.5in}

%\setlength{\oddsidemargin}{1.937cm}  % default -> 2.0 cm odd side
%\setlength{\oddsidemargin}{0.55in} % final version to grad school
\setlength{\oddsidemargin}{0.55in}
\setlength{\evensidemargin}{0.1in} % default -> 3.4 cm even side
%\setlength{\evensidemargin}{0.1in} % default -> 3.4 cm even side

%\setlength{\topmargin}{0.75in}
\setlength{\topmargin}{0.25in}
%\setlength{\headheight}{0.0in}
\setlength{\headheight}{0.4in}
\setlength{\headsep}{0.1in}

\setlength{\marginparsep}{0in}
\setlength{\marginparwidth}{0in}

\setlength{\textwidth}{5.75in}
%\setlength{\textheight}{8.25in}
\setlength{\textheight}{7.85in}

\setlength{\footskip}{0.4in}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Documentstyle Modifications
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\brokenpenalty=10000 % Do not break pages at hyphens.

% use baselinestretch of 2.0 for 10 pt
% use baselinestretch of 1.6 for 11 pt
%
\def\baselinestretch{1.6}
%\@normalsize
%\def\baselinestretch{1.2}

\@addtoreset{equation}{chapter} % Set things up so that the equation,
% thefigure, and thetable counters are reset by incrementing
% thechapter. (The others are correct in the report style.)

% use Table of Contents rather than default of Contents
%\def\tableofcontents {
%  \@restonecolfalse
%  \if@twocolumn
%    \@restonecoltrue\onecolumn
%  \fi
%  \chapter*{Table of Contents\@mkboth{TABLE OF CONTENTS}{TABLE OF CONTENTS}}
%  \@starttoc{toc}
%  \if@restonecol
%    \twocolumn
%  \fi
%}

% single space bibliography, quotations and verse.
\renewenvironment{thebibliography}[1] {
  \chapter*{References}
  \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}
  \list{\@biblabel{\@arabic\c@enumiv}} {
    \settowidth\labelwidth{\@biblabel{#1}}
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \@openbib@code
    \usecounter{enumiv}
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}
  }
  \sloppy
  \clubpenalty4000
  \@clubpenalty \clubpenalty
  \widowpenalty4000
  \sfcode`\.=1000\relax
  %\sfcode`\.\@m
} {
  \def\@noitemerr {
    \@latex@warning{Empty `thebibliography` environment}
  }
  \endlist
}

% This pageheadings will place numbers at top righthand corners
\def\ps@topright {
  \let\@mkboth\@gobbletwo
  \def\@oddhead{\rm\hfil\thepage}
  \let\@evenhead\@oddhead
  \def\@evenfoot{}
  \def\@oddfoot{}
}

%  ***** Verbatim listing macro ****
%  Code from the TeXbook for inputting a file and listing it verbatim in
%  \tt mode. Will not properly handle tabs.  This code hasn't been
%  rigorously tested so it should be used cautiously.  Note that the
%  left quote. "`", is redefined when the macro is finally used or `
%  isn't properly handled.
\def\uncatcodespecials{\def\do##1{\catcode`##1=12 }\dospecials}
\def\setupverbatim {
  \tt \def\par{\leavevmode\endgraf}\catcode`\`=\active
  \obeylines \uncatcodespecials \obeyspaces
} {
  \obeyspaces\global\let =\
}
\def\listing#1{
  \par\begingroup\setupverbatim\input#1
  \endgroup
}

% This is in limbo right now...make heading and subheading as in Scribe.
\def\tempheading{
  \@startsection{donothing}{donothing}{\z@}
  {-3.25ex plus -1ex minus -.2ex}{1.5ex plus .2ex}
  {\centering\large\bf}
}
\def\heading{\tempheading*}
\def\subheading{\subsection*}

% Single Space Figures, Tables, and Footnotes
% Redefine the macro used for floats (including figures and tables) so
% that single spacing is used.  (Note \def\figure{\@float{figure}set
% single spacing} doesn't work because figure has an optional argument)
\def\@xfloat#1[#2] {
  \ifhmode
    \@bsphack\@floatpenalty -\@Mii
  \else
    \@floatpenalty-\@Miii
  \fi
  \def\@captype{#1}
  \ifinner
    \@parmoderr\@floatpenalty\z@
  \else\@next\@currbox\@freelist {
    \@tempcnta\csname ftype@#1\endcsname
    \multiply\@tempcnta\@xxxii\advance\@tempcnta\sixt@@n
    \@tfor \@tempa :=#2\do {
      \if\@tempa h\advance\@tempcnta \@ne\fi
      \if\@tempa t\advance\@tempcnta \tw@\fi
      \if\@tempa b\advance\@tempcnta 4\relax\fi
      \if\@tempa p\advance\@tempcnta 8\relax\fi
    }
    \global\count\@currbox\@tempcnta
  } \@fltovf\fi
  \global\setbox\@currbox\vbox\bgroup
  \def\baselinestretch{1}\@normalsize
  \boxmaxdepth\z@

% leave captions indented, I hope
% \hsize\columnwidth
  \@parboxrestore
}

% Single space footnotes.
\long\def\@footnotetext#1{
  \insert\footins{
    \def\baselinestretch{1}\footnotesize
    \interlinepenalty\interfootnotelinepenalty
    \splittopskip\footnotesep
    \splitmaxdepth \dp\strutbox \floatingpenalty \@MM
    \hsize\columnwidth \@parboxrestore
    \edef\@currentlabel{\csname p@footnote\endcsname\@thefnmark}
    \@makefntext {
      \rule{\z@}{\footnotesep}
      \ignorespaces
      #1\strut
    }
  }
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% User commands
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\draft{\drafttrue}
\def\copyrightpage{\copyrighttrue}
\def\figurespage{\figurestrue}
\def\tablespage{\tablestrue}
\def\phd{
  \phdtrue\def\@degree{Doctor Of Philosophy}
  \def\@initials{Ph.D.}
}
\def\ms{
  \phdfalse\def\@degree{Master Of Science}
  \def\@initials{M.S.}
}
\long\def\abstract#1{\abstracttrue\gdef\@abstract{#1}}
\long\def\preface#1{\prefacetrue\gdef\@preface{#1}}
\long\def\extra#1{\extratrue\gdef\@extra{#1}}
\long\def\acknowledgements#1{\acknowledgementstrue\gdef\@acknowledgements{#1}}
\long\def\dedication#1{\dedicationtrue\gdef\@dedication{#1}}
\def\abbreviations#1{\abbreviationspagetrue\gdef\@abbreviations{#1}}

\def\initials#1{\gdef\@initials{#1}}
\def\degree#1{\phdfalse\gdef\@degree{#1}}
\def\program#1{\gdef\@program{#1}}
\def\words#1{\gdef\@words{#1}}
\def\campus#1{\gdef\@campus{#1}}
\def\director#1{\gdef\@director{#1}}
\def\words#1{\gdef\@words{#1}}
\def\approvals#1{\gdef\@approvals{#1}}
\def\submissionmonth#1{\gdef\@month{#1}}
\def\submissionyear#1{\gdef\@year{#1}}

% defaults
\def\@initials{Ph.D.}
\def\@degree{Doctor of Philosophy}
\def\@campus{University of Minnesota}
\def\@title{}
\def\@author{}
\def\@program{}
\def\@words{34}
\def\@director{}
\def\@words{}
\def\@month{
  \ifcase\the\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
}
\def\@year{\number\the\year}

% conditionals and defaults
\newif\ifdraft			\draftfalse
\newif\ifafterpreface		\afterprefacefalse
\newif\ifcopyright		\copyrightfalse
\newif\iffigures		\figuresfalse
\newif\iftables			\tablesfalse
\newif\ifabstract		\abstractfalse
\newif\ifpreface		\prefacefalse
\newif\ifextra			\extrafalse
\newif\ifacknowledgements	\acknowledgementsfalse
\newif\ifdedication		\dedicationfalse
\newif\ifabbreviationspage	\abbreviationspagefalse
\newif\ifphd			\phdfalse
\newif\ifpagestyletopright	\pagestyletoprighttrue

\newcommand\titlep {
  \vbox to \textheight {
    \parskip0pt
    \parindent0pt
    \typeout{Title}
    \null\nobreak\vskip 10bp {
      \def\baselinestretch{1}\@normalsize
      \centering\Large\bf\uppercase\expandafter{\bf\@title}
      \par\nobreak
    }
    \vskip 4ex plus 2ex minus 2ex {
      \centering\bf
      \ifdraft {
        DRAFT VERSION CREATED ON \uppercase\expandafter{\today}\\
      }\else 
        A \ifphd{DISSERTATION}\else{THESIS}\fi\ \\
        SUBMITTED TO THE FACULTY OF THE GRADUATE SCHOOL \\
        OF THE UNIVERSITY OF MINNESOTA \\
        BY \\
% probably want to add space.
% \vskip 36bp minus 36bp
        \vskip 2ex plus 2ex minus 2ex
        \@author\par
        \nobreak
        \vskip 2ex plus 2ex minus 2ex
        IN PARTIAL FULFILLMENT OF THE REQUIREMENTS \\
        FOR THE DEGREE OF\\
%       \ifphd{Doctor of Philosophy}\else{Master of Science}\fi\ \\
        \ifphd{Doctor of Philosophy}\else \@degree          \fi\ \\
        \vskip 1ex plus 1ex minus 1ex
        \@director\par
        \vskip 1ex plus 1ex minus 1ex
        \@month, \@year\\
      \fi %close ifdraft
    }
  }
  \newpage
}% end titlep

\newcommand\copyrightp {
  \typeout{Copyright}
  \null\vfil
  \begin{center}
  \bf\large\copyright\ \@author\ \ \@year\ \\
  ALL RIGHTS RESERVED
  \end{center}
  \vfil\newpage
}

\def\desclabel#1{{\bf#1\hfill}}
\def\descriptionlist#1 {
  \list{} {
    \settowidth\labelwidth{#1}
    \leftmargin\labelwidth
    \let\makelabel\desclabel
  }
}
\let\enddescriptionlist=\endlist

\def\unnumberedsection#1 {
  \chapter*{#1}
  \addcontentsline{toc}{chapter}{#1}
  \typeout{#1}
}

% The signature, title, copywrite, and abstract pages
\newcommand\beforepreface {
%  \ifabstract {
  \ifacknowledgements {  
    \c@page1 % set page counter so that next is 1
    \pagestyle{empty}


% Title page
    \titlep

% Copywrite page
    \ifcopyright\copyrightp\fi

%Acknowledgements 

  \pagenumbering{roman}    
  \pagestyle{plain} 
  \ifacknowledgements
  
    \unnumberedsection{Acknowledgements}

%    \addcontentsline{toc}{chapter}{Acknowledgements}
%    \null\vskip -20bp
%    \begin{center} {
%    }
%    \large\bf\
%    Acknowledgements
%    \end{center}
%    \vskip 8bp

    \@acknowledgements\fi   
  \vfil\newpage  
  }  

%Dedication

% \pagenumbering{roman}     
  \ifdedication
  
    \unnumberedsection{Dedication}

%    \addcontentsline{toc}{chapter}{Dedication}
%    \null\vskip -20bp
%    \begin{center} {
%    }
%    \large\bf\
%    Dedication
%    \end{center}
%    \vskip 8bp

    \@dedication\fi   
  \vfil\newpage  
    


% An abstract to be bound into the thesis
%    \pagestyle{plain}
%    \pagenumbering{roman}
%    \addcontentsline{toc}{chapter}{Abstract}
%    \null\vskip -20bp
%    \begin{center} {
%      \large\bf\expandafter{\@title} \large
%      \vskip 8bp
%      by \@author \\
%      %Under the supervision of \@director
%      \vskip 8bp
%    }
%    \large\bf\
%    Abstract
%    \end{center}
%    \vskip 8bp
%    \@abstract
%  } \fi
%  \typeout{Abstract}
%  %\eject
}


\newcommand\afterpreface {
% Insert all of the text between the title (and copyright) pages and the main
% body of the thesis.
%  \pagenumbering{roman}
%  \ifpreface {
%    \unnumberedsection{Preface}
%    \@preface
%  } \fi
%  \ifacknowledgements
%    \unnumberedsection{Acknowledgements}
%    \@acknowledgements
%  \fi
%  \ifdedication 
%    \unnumberedsection{Dedication}
%    \@dedication
%   \fi

    \pagestyle{plain}
%    \pagenumbering{roman}
    \ifabstract {
    \addcontentsline{toc}{chapter}{Abstract}
    \null\vskip -20bp
    \begin{center} {
    }
    \large\bf\
    Abstract
    \end{center}
    \vskip 8bp
    \@abstract
   \fi
  \typeout{Abstract}
}

  \tableofcontents
  \iftables
    \addvspace{10bp}
    \let\saveaddvspace=\addvspace
    \def\addvspace##1{}
    \listoftables
    \let\addvspace=\saveaddvspace
  \fi
  
 
  \iffigures
    \addvspace{10bp}
    \let\saveaddvspace=\addvspace
    \def\addvspace##1{}
    \listoffigures
    \let\addvspace=\saveaddvspace
  \fi
  
  \ifabbreviationspage {
    \def\baselinestretch{1}
    \@normalsize
    \unnumberedsection{List of Abbreviations}
    \@abbreviations
  } \fi
  \ifextra {
%   \chapter*{}
    \typeout{extra}
    \newpage
    \@extra
  } \fi
  \newpage
  \afterprefacetrue
  \c@page0 % set page counter so that next is 1
  \pagenumbering{arabic}
  \pagestyle{topright}
}

\def\vita {
  \unnumberedsection{Vita}
  \bgroup\def\baselinestretch{1}
  \@normalsize
}

\def\endvita{\egroup}

% Fullpage floats:
\def\fullpagefigure {
  \@float{figure}[p]
  \vbox to \textheight
  \bgroup
}
\def\endfullpagefigure{\egroup\end@float}

\def\fullpagetable {
  \@float{table}[p]
  \vbox to \textheight
  \bgroup
}
\def\endfullpagetable{\egroup\end@float}

% Redefine appendix to print Appendix []. ... page # in table of contents.
\def\appnumberline#1{
  \advance\hangindent\@tempdima
  \hbox{Appendix #1. }
}

\renewcommand\appendix {
  \par
  \setcounter{chapter}{0}
  \setcounter{section}{0}
  \def\@chapapp{Appendix}
  \def\thechapter{\Alph{chapter}}
% Redefine @chapter  so that appnumberline (ratherline) is used.
  \def\@chapter[##1]##2 {
    \ifnum \c@secnumdepth >\m@ne
      \refstepcounter{chapter}
      \typeout{\@chapapp\space\thechapter.}
      \addcontentsline{toc}{chapter}
      {\protect \appnumberline{\thechapter}##1}
    \else
      \addcontentsline{toc}{chapter}{##1}
    \fi
    \chaptermark{##1}
% Don't add extra space to lists of figures and tables after chapters.
 \addtocontents{lof}{\protect\addvspace{10pt}}
 \addtocontents{lot}{\protect\addvspace{10pt}}
    \if@twocolumn
      \@topnewpage[\@makechapterhead{##2}]
    \else
      \@makechapterhead{##2}
      \@afterheading
    \fi
  }
}


\renewcommand\listoffigures {
  \@restonecolfalse
  \if@twocolumn
    \@restonecoltrue
    \onecolumn
  \fi
  \chapter* {
    List of Figures
    \@mkboth{LIST OF FIGURES}{LIST OF FIGURES}
  }
  \addcontentsline{toc}{chapter}{List of Figures}
  \@starttoc{lof}
  \if@restonecol\twocolumn\fi
}

\renewcommand\listoftables {
  \@restonecolfalse
  \if@twocolumn
    \@restonecoltrue
    \onecolumn
  \fi
  \chapter* {
    List of Tables
    \@mkboth{LIST OF TABLES}{LIST OF TABLES}
  }
  \addcontentsline{toc}{chapter}{List of Tables}
  \@starttoc{lot}
  \if@restonecol\twocolumn\fi
}

\renewcommand{\bibname}{References} 
\renewenvironment{thebibliography}[1]
     {\chapter*{\bibname}%
      \addcontentsline{toc}{chapter}{\bibname}
      \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
