export const NoticeBanner = () => {
  return (
    <div className="banner-notice" role="alert">
      <strong>Demo Notice:</strong>
      <span>
        This prototype classifies customer support tickets into CFPB financial product categories (run_004 baseline). It
        does not detect out-of-domain inquiries or non-financial support requests. Ticket priority prediction is deferred.
      </span>
    </div>
  );
};

